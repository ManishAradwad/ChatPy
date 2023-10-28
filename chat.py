from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
from langchain.prompts import MessagesPlaceholder
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool

os.environ[
    "SERPAPI_API_KEY"
] = "<your-api-key>"

openai_api = "<your-api-key>"
llm = ChatOpenAI(
    openai_api_key=openai_api,
    temperature=0.9,
    model_name="gpt-4",
)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    )
]
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

while True:
    prompt = input("Enter your prompt: ")
    if not prompt or prompt == "q":
        print("Goodbye!")
        break

    print(agent.run(prompt))
