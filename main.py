# Start from the last coding stage of the previous LLM evals project
import json
import os
import sys
import uuid

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import observe, propagate_attributes, get_client
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables from .env file
dotenv.load_dotenv()

# Generate unique session_id and user_id once
session_id = f"session-{uuid.uuid4().hex[:8]}"
users = ["James", "George", "Mike", "Sherlock"]
user_id = users[uuid.uuid4().int % len(users)]

# Initialize the LLM with OpenAI API credentials (substitute for other models)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True
)

# Initialize conversation history
conversation = []

# Initialize Langfuse client
langfuse = get_client()


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

@observe(name="embed_documents")
def embed_documents(json_path: str) -> QdrantVectorStore | list:
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        A Qdrant vector store built from the smartphone documents,
        or an empty list if an error occurs.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return []
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON from file {json_path}: {jde}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_path}: {e}")
        return []

    documents = []
    for entry in data:
        # Build a readable content string from the JSON entry
        content = (
            f"Model: {entry.get('model', '')}\n"
            f"Price: {entry.get('price', '')}\n"
            f"Rating: {entry.get('rating', '')}\n"
            f"SIM: {entry.get('sim', '')}\n"
            f"Processor: {entry.get('processor', '')}\n"
            f"RAM: {entry.get('ram', '')}\n"
            f"Battery: {entry.get('battery', '')}\n"
            f"Display: {entry.get('display', '')}\n"
            f"Camera: {entry.get('camera', '')}\n"
            f"Card: {entry.get('card', '')}\n"
            f"OS: {entry.get('os', '')}\n"
            f"In Stock: {entry.get('in_stock', '')}"
        )
        documents.append(Document(page_content=content))

    try:
        collection_name = "smartphones"
        qdrant_client = QdrantClient("http://localhost:6333")

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )

            qdrant_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings_model
            )

            qdrant_store.add_documents(documents=documents)

            return qdrant_store

        # no need to create a vector store every time
        else:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings_model,
                collection_name=collection_name,
                url="http://localhost:6333"
            )

            return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


# Initialize the vector store
product_db = embed_documents("datasets/smartphones.json")


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieves information about a smartphone model from the product database.

    :param
        model (str): The smartphone model to search for.

    :returns
        str: The smartphone's specifications, price, and availability,
             or an error message if not found or if an error occurs.
    """
    try:
        results = product_db.similarity_search(model, k=1)
        if not results:
            print(f"Info: No results found for model: {model}")
            return "Could not find information for the specified model."
        info = results[0].page_content
        return info
    except Exception as e:
        return f"Error during smartphone information retrieval for model {model}: {e}"


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
@observe(name="generate_context")
def generate_context(ai_message: AIMessage, current_conversation: list) -> None:
    """
    Process tool calls from the language model and append the AI message and
    each tool's response as ToolMessage objects to the conversation history.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.
        current_conversation (list): The per-turn scratch conversation to augment with tool context.
    """
    # Add tool-call context to the per-turn scratch conversation.
    # Redis stores only clean HumanMessage/AIMessage pairs; tool messages are not persisted.
    current_conversation.append(ai_message)

    # Check if the AI message has any tool calls
    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        current_conversation.append(
            AIMessage(
                content="No tool calls found. Please ensure the model is configured to use tools."
            )
        )

    try:
        # Process each tool call, invoke the appropriate tool, and append the result to the conversation
        # a message with tool calls is expected to be followed by tool responses
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] == "SmartphoneInfo":
                tool_output = smartphone_info_tool.invoke(tool_call)
                current_conversation.append(tool_output)

    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        current_conversation.append(
            AIMessage(
                content=f"An error occurred while processing tool calls: {e}"
            )
        )


# ---------------------------
# Main Conversation Loop
# ---------------------------
def main():
    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    context_lf_prompt = langfuse.get_prompt("context_system_prompt")
    context_prompt = ChatPromptTemplate.from_messages([
        *context_lf_prompt.get_langchain_prompt()
    ])
    context_prompt.metadata = {"langfuse_prompt": context_lf_prompt}

    review_lf_prompt = langfuse.get_prompt("review_system_prompt")
    review_prompt = ChatPromptTemplate.from_messages([
        *review_lf_prompt.get_langchain_prompt()
    ])
    review_prompt.metadata = {"langfuse_prompt": review_lf_prompt}

    goodbye_lf_prompt = langfuse.get_prompt("goodbye_system_prompt")
    goodbye_prompt = ChatPromptTemplate.from_messages(
        goodbye_lf_prompt.get_langchain_prompt()
    )
    goodbye_prompt.metadata = {"langfuse_prompt": goodbye_lf_prompt}

    context_chain = context_prompt | llm_with_tools | (lambda ai_message: generate_context(ai_message, conversation))
    review_chain = review_prompt | llm

    goodbye_chain = goodbye_prompt | llm

    redis_history = RedisChatMessageHistory(
        session_id=session_id,
        redis_url=os.getenv("REDIS_URL"),
        ttl=3600,
    )

    # Initialize the Langfuse handler once for the entire conversation
    langfuse_handler = CallbackHandler()

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "end"]:
                # Create a parent span for the goodbye message
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="user-query",
                    input={"user_input": user_input}
                ) as span:
                    with propagate_attributes(
                        session_id=session_id,
                        user_id=user_id
                    ):
                        goodbye_message = goodbye_chain.invoke(
                            {"user_id": user_id},
                            config={
                                "run_name": "goodbye-message",
                                "callbacks": [langfuse_handler]
                            }
                        )

                        # Set the output on the parent span
                        span.update(output={"response": goodbye_message.content})

                print(f"System: {goodbye_message.content}")

                # Collect user feedback about the entire conversation
                feedback = input("\nWas this conversation helpful? (Yes/No): ").strip()
                user_comment = input("Please give us a reason for your answer. This will help us improve: ").strip()

                # Score at the session level (not individual trace)
                langfuse.create_score(
                    session_id=session_id,  # Use the session_id from the start of the conversation
                    name="conversation_usefulness",
                    value=feedback,
                    data_type="CATEGORICAL",
                    comment=user_comment
                )

                print("\nThank you for your feedback!")
                langfuse.flush()
                break

            user_message = HumanMessage(content=user_input)
            history = list(redis_history.messages)
            prompt_conversation = trim_messages(
                history + [user_message],
                max_tokens=1000,
                strategy="last",
                token_counter=llm,
                include_system=True,
                allow_partial=False,
            )

            conversation.clear()
            conversation.extend(prompt_conversation)

            # Create a parent span for this user query to group all chain invocations
            with langfuse.start_as_current_observation(
                as_type="span",
                name="user-query",
                input={"user_input": user_input}
            ) as span:
                # Propagate trace attributes to all child observations
                with propagate_attributes(
                    session_id=session_id,
                    user_id=user_id
                ):
                    # Context chain invocation
                    context_chain.invoke(
                        {"user_input": user_input, "conversation": conversation},
                        config={
                            "run_name": "context",
                            "callbacks": [langfuse_handler]
                        }
                    )

                    # Final response chain invocation
                    response = review_chain.invoke(
                        {"user_id": user_id, "user_input": user_input, "conversation": conversation},
                        config={
                            "run_name": "final-response",
                            "callbacks": [langfuse_handler]
                        }
                    )

                    redis_history.add_message(user_message)
                    redis_history.add_message(response)

                # Set the output on the parent span
                span.update(output={"response": response.content})

            print(f"System: {response.content}")
            langfuse.flush()

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        langfuse.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
