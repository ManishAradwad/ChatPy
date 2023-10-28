from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
from langchain.prompts import MessagesPlaceholder
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool

os.environ[
    "SERPAPI_API_KEY"
] = "c8c67651978e5d3bb3cf305931d3a9506f4ece0a7948dd73afce9ce75f825655"
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

llm = ChatOpenAI(
    openai_api_key="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",
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
