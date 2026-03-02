from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()

def main():
    print("Hello!")

    information = """Alex Sandro Silva Pereira[1] (born 7 July 1987) is a Brazilian professional mixed martial artist and former professional kickboxer. He currently competes in the Light Heavyweight division of the Ultimate Fighting Championship (UFC), where he is the current and two-time UFC Light Heavyweight Champion and former UFC Middleweight Champion. As of 3 February 2026, he is #5 in the UFC men's pound-for-pound rankings.[8]

He is the ninth fighter in UFC history to become champion in two different weight divisions and the first to become champion in both the middleweight and light heavyweight divisions of the organization. In kickboxing, he is a former Glory middleweight and light heavyweight champion, and is the first and only fighter to have held Glory titles in two weight classes simultaneously. Pereira also competed in promotions such as It's Showtime and Superkombat Fighting Championship in kickboxing, and for Jungle Fight and Legacy Fighting Alliance in MMA. Pereira is the only known fighter to be a two-division world champion in MMA and kickboxing and is regarded as one of the greatest combat athletes of all time.[9][10][11][12][13] Pereira was ranked #1 in the kickboxing middleweight and light-heavyweight rankings in February
"""
    summary_template = """
    given the information {information} about a person I want you create:
    1. A short summary
    2. two interesting fact about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ['infomation'], template = summary_template
    )

    #llm = ChatOpenAI(temperature = 0, model = "gpt-5")
    llm = ChatOllama(temperature =0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)



if __name__ == "__main__":
    main()
