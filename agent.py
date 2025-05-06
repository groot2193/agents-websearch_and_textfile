import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai



#you get all thsi from the documentation agno and phidata
web_search_agent=Agent(
    name="Web search agent",
    role="search the web for the information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)


finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama3-70b-8192"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)
multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVIDIA", stream=True)




#use streamlit to add a frontend
#also in case you forgets its a agentic ai and im using agno for it and basically it does web searcha dn figure out things
#as in if i give in a prompt it helps me by doign websearch


st.set_page_config(page_title="AI Multi-Agent System", layout="wide")

st.title("AI Multi-Agent System")
st.write("This system can search the web and fetch financial data.")

# User input
user_input = st.text_area("Enter your query:", "Summarize analyst recommendations and share the latest news for NVIDIA")

# Button to generate response
if st.button("Generate Response"):
    with st.spinner("Processing..."):
        response = multi_ai_agent.run(user_input, stream=True)
        st.markdown(response)  # Display AI response in markdown format
