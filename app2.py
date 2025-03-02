from config import set_environment

set_environment()

import streamlit as st
import time
import random
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, load_tools, create_react_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


def load_agent() -> AgentExecutor:
    llm = ChatOpenAI(temperature=0.2, streaming=True)

    # Create a basic DuckDuckGo search wrapper with default parameters
    ddg_search = DuckDuckGoSearchAPIWrapper()

    # Load most tools normally
    tools = load_tools(tool_names=["arxiv", "wikipedia", "wolfram-alpha"], llm=llm)

    # Add DDG search tool with our custom wrapper
    from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

    ddg_tool = DuckDuckGoSearchRun(api_wrapper=ddg_search)
    tools.append(ddg_tool)

    prompt = PromptTemplate(
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        template=(
            "You are an assistant with access to the following tools: {tool_names}\n\n"
            "{tools}\n\n"
            "Use the following format *EXACTLY*:\n"
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
    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)


# Create a custom wrapper for the search engine with retry logic
def invoke_with_retry(chain, input_data, callbacks):
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data, callbacks=callbacks)
        except Exception as e:
            if "RatelimitException" in str(e) and attempt < max_retries - 1:
                # Add jitter to the delay to prevent synchronized retries
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                st.warning(
                    f"Rate limit hit. Waiting {delay:.1f} seconds before retry..."
                )
                time.sleep(delay)
            else:
                if "RatelimitException" in str(e):
                    st.error("Search rate limit exceeded. Please try again later.")
                    return {
                        "output": "I'm sorry, but the search service is currently rate-limited. Please try a different query or try again later."
                    }
                else:
                    raise e


chain = load_agent()

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())

        try:
            response = invoke_with_retry(
                chain, {"input": prompt}, callbacks=[st_callback]
            )

            if response:
                print(f"Response: {response}")  # print the whole response.
                if (
                    "agent_outcome" in response
                    and response["agent_outcome"] is not None
                ):
                    print(f"Agent Outcome: {response['agent_outcome']}")
                st.write(response["output"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
