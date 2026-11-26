
import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. Please check your .env file."
    )

if not SERPER_API_KEY:
    raise ValueError(
        "SERPER_API_KEY is missing. Please check your .env file."
    )

print("\nAPI keys loaded successfully!")
print(f"Using model: {MODEL}")


# ==================================================
# 2. INITIALIZE SEARCH TOOL
# ==================================================

search_tool = SerperDevTool()


# ==================================================
# 3. CREATE RESEARCHER AGENT
# ==================================================

researcher = Agent(
    role="Technology News Researcher",

    goal=(
        "Find today's top 3 important technology news "
        "stories from reliable online sources."
    ),

    backstory=(
        "You are an experienced technology journalist "
        "specializing in Artificial Intelligence, technology, "
        "startups, cybersecurity and digital innovation. "
        "You identify important news and provide accurate "
        "and concise summaries."
    ),

    tools=[search_tool],

    llm=MODEL,

    verbose=True
)


# ==================================================
# 4. CREATE WRITER AGENT
# ==================================================

writer = Agent(
    role="Technology Newsletter Writer",

    goal=(
        "Convert research findings into a clear, interesting "
        "and professional five-line technology newsletter."
    ),

    backstory=(
        "You are a professional newsletter editor. "
        "You transform complex technology news into simple "
        "and engaging language. You use only the research "
        "provided by the Researcher and never conduct "
        "additional searches."
    ),

    llm=MODEL,

    verbose=True
)


# ==================================================
# 5. CREATE RESEARCH TASK
# ==================================================

find_news = Task(

    description=(
        "Search the internet for today's top 3 technology "
        "news stories.\n\n"

        "Focus on:\n"
        "1. Artificial Intelligence\n"
        "2. Technology companies\n"
        "3. Cybersecurity\n"
        "4. Startups and innovation\n"
        "5. Science and digital technology\n\n"

        "For each news story provide:\n"
        "- News headline\n"
        "- Short factual summary\n"
        "- Source name\n"
        "- Publication date\n"
        "- Source URL\n\n"

        "Use recent and reliable sources. "
        "Do not invent facts or publication dates."
    ),

    expected_output=(
        "A structured research report containing exactly "
        "3 recent technology news stories. Each story must "
        "include its headline, summary, source name, "
        "publication date and source URL."
    ),

    agent=researcher
)


# ==================================================
# 6. CREATE NEWSLETTER WRITING TASK
# ==================================================

write_newsletter = Task(

    description=(
        "Use the Researcher's report to create a professional "
        "technology newsletter.\n\n"

        "Follow this exact five-line format:\n\n"

        "Line 1: Tech in 5 Lines (heading)\n"
        "Line 2: First technology news story\n"
        "Line 3: Second technology news story\n"
        "Line 4: Third technology news story\n"
        "Line 5: Why it matters (one concise sentence)\n\n"

        "Use simple English and informative sentences.\n"
        "Do not conduct additional web searches.\n"
        "Use only the Researcher's findings.\n"
        "Do not add an introduction or conclusion."
    ),

    expected_output=(
        "Exactly 5 lines:\n"
        "Line 1: Tech in 5 Lines\n"
        "Line 2: First news story\n"
        "Line 3: Second news story\n"
        "Line 4: Third news story\n"
        "Line 5: Why it matters\n"
        "No additional lines."
    ),

    agent=writer,

    context=[find_news]
)


# ==================================================
# 7. ASSEMBLE THE CREW
# ==================================================

tech_news_crew = Crew(

    agents=[
        researcher,
        writer
    ],

    tasks=[
        find_news,
        write_newsletter
    ],

    process=Process.sequential,

    verbose=True
)


# ==================================================
# 8. EXECUTE THE CREW
# ==================================================

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("       TECH NEWSLETTER CREW STARTED")
    print("=" * 50)

    try:

        result = tech_news_crew.kickoff()

        print("\n" + "=" * 50)
        print("          FINAL TECH NEWSLETTER")
        print("=" * 50 + "\n")

        print(result)

        print("\n" + "=" * 50)
        print("       CREW EXECUTION COMPLETED")
        print("=" * 50)

    except Exception as error:

        print("\nERROR OCCURRED:")
        print(error)
