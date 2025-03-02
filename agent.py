"""Defining an agent."""

from typing import Literal

import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_openai import ChatOpenAI

from src.tool_loader import load_tools
from config import set_environment
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)  # Ensure this is correctly implemented.

set_environment()

ReasoningStrategies = Literal["zero-shot-react", "plan-and-solve"]


def load_agent(
    tool_names: list[str], strategy: ReasoningStrategies = "zero-shot-react"
) -> LLMChain:
    llm = ChatOpenAI(temperature=0, streaming=True)
    tools = load_tools(tool_names=tool_names, llm=llm)
    if strategy == "plan-and-solve":
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        return PlanAndExecute(planner=planner, executor=executor, verbose=True)

    prompt = hub.pull("hwchase17/react")
    return AgentExecutor(
        agent=create_react_agent(llm=llm, tools=tools, prompt=prompt), tools=tools
    )


st.title("🔍 📚 🎓 Research Assistant")
st.write(
    "Ask me anything and I'll search Wikipedia and academic papers to find answers."
)
strategy = st.radio("Reasoning strategy", ("plan-and-solve", "zero-shot-react"))

tool_names = st.multiselect(
    "Which tools do you want to use?",
    [
        "google-search",
        "ddg-search",
        "wolfram-alpha",
        "arxiv",
        "wikipedia",
        "pal-math",
        "llm-math",
    ],
    ["ddg-search", "wolfram-alpha", "wikipedia"],
)

agent_chain = load_agent(tool_names=tool_names, strategy=strategy)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load the agent
try:
    chain = agent_chain  # corrected line
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
