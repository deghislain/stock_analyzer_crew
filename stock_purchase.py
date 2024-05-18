# Warning control
import warnings

warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from crewai import Crew, Process
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

data_collector_agent = Agent(
    role="Data Collector",
    goal="Collect financial data from the internet, "
         "data needed to predict the performance of a given stock."
         "Only use website that allow access to the needed information.",
    backstory="Specializing in financial markets, this agent "
              "scrap financial websites to retrieve information about a given stock, "
              "the Data Collector Agent is the cornerstone for "
              "collecting all kind of information needed to predict, "
              "the performance of a given stock.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Analyse stock and company information to predict stock performance"
         "based on data collected by the Data Collector Agent.",
    backstory="Specializing in financial markets, this agent "
              "uses statistical modeling and machine learning "
              "to provide crucial insights. With a knack for data, "
              "the Data Analyst Agent is the cornerstone for "
              "informing trading decisions.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks "
         "associated with the purchase of a given stock.",
    backstory="Armed with a deep understanding of risk assessment models "
              "and market dynamics, this agent scrutinizes the potential "
              "risks of proposed stocks. It offers a detailed analysis"
              "of the potential growth of a given stock.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)
stock = input("Insert stock symbol:")
docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://ca.finance.yahoo.com/quote/{stock}/financials"
)
data_collector_task = Task(
    description=(
        "Scrap the content of the most important financial websites "
        "to extract financial data related to"
        "the selected stock ({stock_selection}). "
    ),
    expected_output=(
        "Financial data for {stock_selection}."
    ),
    tools=[docs_scrape_tool],
    agent=data_collector_agent,
)

data_analysis_task = Task(
    description=(
        "Analyse company's financial data based on "
        "financial data provided by the Data Collector and"
        "the selected stock ({stock_selection}). "
        "Use statistical modeling and machine learning to "
        "predict the performance of the given stock."
        "Do not search the internet."
    ),
    expected_output=(
        "A comprehensive data analysis report detailing "
        "the market performance of {stock_selection}."
    ),
    agent=data_analyst_agent,
)

risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the purchase"
        "of the {stock_selection}. "
        "Provide a detailed analysis of potential risks "
        "and make suggestion about the purchase of a stock,"
        "based on the data analysis report provided by the Data Analyst."
        "Do not search the internet."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential "
        "risks related to the purchase of {stock_selection}."
    ),
    agent=risk_management_agent,
)

financial_trading_crew = Crew(
    agents=[data_collector_agent,
            data_analyst_agent,
            risk_management_agent],

    tasks=[data_collector_task,
           data_analysis_task,
           risk_assessment_task],

    manager_llm=ChatOpenAI(model="gpt-3.5-turbo",
                           temperature=0.7),
    process=Process.hierarchical,
    verbose=True,
    memory=True
)

financial_trading_inputs = {
    'stock_selection': stock
}

result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)

print("*****************************************************************************************", stock)
print(result)
