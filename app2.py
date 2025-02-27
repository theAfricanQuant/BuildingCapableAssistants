from config import set_environment

set_environment()

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, load_tools, create_react_agent
from langchain_community.chat_models import ChatOpenAI


def load_agent() -> AgentExecutor:
    llm = ChatOpenAI(temperature=0.2, streaming=True)
    tools = load_tools(tool_names=["arxiv", "wikipedia"], llm=llm)

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
    return AgentExecutor(agent=agent, tools=tools)


chain = load_agent()

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = chain.invoke(
            {"input": prompt}, callbacks=[st_callback], handle_parsing_errors=True
        )
        print(f"Response: {response}")  # print the whole response.
        if "agent_outcome" in response and response["agent_outcome"] is not None:
            print(f"Agent Outcome: {response['agent_outcome']}")
        st.write(response["output"])
