from langchain.agents import AgentExecutor, load_tools, create_react_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import time
from config import set_environment

set_environment()


def load_agent() -> AgentExecutor:
    llm = ChatOpenAI(temperature=0, streaming=True)

    # Change the tools configuration to use more reliable tools
    # Wikipedia and arXiv are less likely to hit rate limits than DuckDuckGo
    tools = load_tools(
        tool_names=["wikipedia", "arxiv"],  # Remove ddg-search to avoid rate limits
        llm=llm,
    )

    # Define a valid ReAct-style prompt with required placeholders
    prompt = PromptTemplate(
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        template=(
            "You are an assistant with access to the following tools: {tool_names}\n\n"
            "{tools}\n\n"
            "Use the following format:\n"
            "Question: the input question\n"
            "Thought: consider what to do\n"
            "Action: the action to take, should be one of {tool_names}\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Question: {input}\n"
            "{agent_scratchpad}"
        ),
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


st.title("🔍 📚 🎓 Research Assistant")
st.write(
    "Ask me anything and I'll search Wikipedia and academic papers to find answers."
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load the agent
try:
    chain = load_agent()
except Exception as e:
    st.error(f"Error loading agent: {e}")
    st.stop()

# Handle user input
if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        response_container = st.container()
        st_callback = StreamlitCallbackHandler(response_container)

        try:
            # Use a try-except block to handle potential errors
            response = chain.invoke({"input": prompt}, callbacks=[st_callback])
            output = response.get(
                "output", "I couldn't generate a response. Please try again."
            )
            st.markdown(output)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": output})

        except Exception as e:
            error_msg = str(e)
            if "RatelimitException" in error_msg:
                st.error(
                    "I hit a rate limit with the search tools. Please try again in a minute or ask a different question."
                )
            else:
                st.error(f"Error: {error_msg}")
