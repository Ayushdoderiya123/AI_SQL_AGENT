import streamlit as st 
import sqlite3
import os
import pandas as pd
import re
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph,START,END,add_messages
from langchain_core.messages import AnyMessage,AIMessage,HumanMessage
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from typing import TypedDict,List,Annotated
import json
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManager
from langchain_community.callbacks.context_callback import ContextCallbackHandler
from numpy.f2py.crackfortran import previous_context

# Load .env file
load_dotenv()
st.title("AI SQL AGENT")
MESSAGE_LIMIT = 5
MAX_MESSAGE_LENGTH = 300
token = "8F8TGoThVcv2DnBB3UnoSTyz"
callback=ContextCallbackHandler(token)
callback_manager = CallbackManager([callback])
def trim_messages(messages):
    """Keep only the last MESSAGE_LIMIT messages and truncate long messages."""
    trimmed_messages = messages[-MESSAGE_LIMIT:]  # Keep last N messages
    # print(f"{trimmed_messages=}\n")
    for msg in trimmed_messages:
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            msg.content = msg.content[-MAX_MESSAGE_LENGTH:] # Truncate long messages
            print(f"{msg.content=}\n")
    return trimmed_messages

def infer_sqlite_type(dtype):
    """Map pandas dtypes to SQLite types."""
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "REAL"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TEXT"  # Store dates as ISO strings
    elif pd.api.types.is_object_dtype(dtype):
        return "TEXT"
    else:
        return "TEXT"

def upload_csv_to_sqlite(db_path, table_name, csv_file):
    """Automatically creates a table and uploads CSV data to SQLite without duplication."""
    if ".xlsm"  in csv_file.name:
        df = pd.read_excel(csv_file)
        df[df.select_dtypes(include=[object]).columns] = df.select_dtypes(include=[object]).astype('string')
        df[df.select_dtypes(include=["datetime64[ns]"]).columns]=df.select_dtypes(include=["datetime64[ns]"]).astype("string")
    if ".csv" in csv_file.name:
        df = pd.read_csv(csv_file)

    st.write(df.head())
    # df = df.astype(str)
    print(f"{df.info()=}")
    columns= []
    # Infer column types
    column_types = {col: infer_sqlite_type(df[col]) for col in df.columns  }

    print(f"{column_types=}/n")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension("stats.dll")
    cursor = conn.cursor()

    # Create table dynamically if it doesn't exist
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    create_table_sql += ", ".join(f'"{col}" {col_type}' for col, col_type in column_types.items())
    # Check if the dataset has an ID column to set as PRIMARY KEY
    if "id" in df.columns:
        create_table_sql += ', PRIMARY KEY("id")'  # Add primary key constraint

    create_table_sql += ");"
    print(f"{create_table_sql=}\n")
    cursor.execute(create_table_sql)
    conn.commit()

    # Insert data without duplicates
    placeholders = ", ".join(["?"] * len(df.columns))
    # columns = ", ".join(df.columns)
    columns = ", ".join(f'"{col}"' for col in df.columns)
    print(f"{columns=}")
    insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    print(f"{insert_sql=}")
    # Convert DataFrame to list of tuples
    data = [tuple(row) for row in df.itertuples(index=False, name=None)]

    # print(f"{data=}")
    # Check if the table already has data
    print("t")
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    print("t")
    row_count = cursor.fetchone()[0]
    print(f"{row_count=}")
    if row_count == 0:
       cursor.executemany(insert_sql, data)  # Insert without duplicates
       conn.commit()

    st.write(f"Successfully uploaded file into table '{table_name}' in {db_path}")

    conn.close()


def check_tables_in_sqlite(db_path):
    """Check if the table exists in SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return [table[0] for table in tables]

# Example Usage
db_file = "my_database.db"
table_name = "my_table"

# Use Streamlit's file uploader
csv_file = st.file_uploader("Upload the CSV file")
print(f"{csv_file=}")
if csv_file is not None:
    upload_csv_to_sqlite(db_file, table_name, csv_file)
    # model1 = init_chat_model("llama3-8b-8192",model_provider="groq")
    model1 = init_chat_model("gemini-2.0-flash",model_provider ="google_genai")
    db = SQLDatabase.from_uri("sqlite:///my_database.db")
    toolkit = SQLDatabaseToolkit(db=db,llm=model1)
    tools = toolkit.get_tools()
    tool_list = [tool for tool in tools]
    print(f"{tool_list=}\n")
    query_check = [tool for tool in tools if tool.name == "sql_db_query_checker"][0]
    print(f"{query_check=}\n")
    sql_db_query = [tool for tool in tools if tool.name =="sql_db_query"][0]
    print(f"{sql_db_query=}\n")
    db_schema =  db.get_table_info()

    # print(f"{db_schema=}")

else:
    st.write("Please upload a CSV file to continue.")
class State(TypedDict):
  messages:Annotated[list[AnyMessage],add_messages]


sql_prompt = PromptTemplate(
    input_variables=["question", "schema","history"],
    template="""
    You are an AI that converts natural language questions into SQL queries first read and analyze schema properly now if any question asked by the user you should be able to answer like column names,correlation,covariance etc.
    You know very well how to convert any question asked by the user in NLP to SQL no matter how complex or easy question asked by the user you should be able to convert it into a correct SQL query without any logical mistake and syntax mistake.You should be writing table name and column name correct while creating queries.
    Ensure the SQL query only uses columns from the provided schema and know everything about all column with their name, data type and other information.

    Listen one important thing also if user prompt is based on previous user prompt context and if previous prompt is there in history generates SQL queries based by remembering the previous context which is able to generate accurate output . Ensure the SQL query only uses columns from the provided schema it is must.

    Database Schema:
    {schema}

    Question: {question}

    History : {history}


    Respond **only** with a valid JSON object containing the SQL query. Do not include explanations, notes, or any extra text.

    JSON Output Format:
    {{
        "sql_query": "<SQL_QUERY_HERE>"
    }}
    """
)

# model2 = init_chat_model("Llama3-8b-8192", model_provider="groq")
model2 = init_chat_model("gemini-2.0-flash",model_provider ="google_genai",callback_manager = callback_manager)
sql_chain = LLMChain(llm=model2, prompt=sql_prompt)
print(f"{sql_chain=}\n")
conversational_history = []
def convert_to_sql(state: State) -> State:
    global conversational_history
    user_question = state["messages"][-1].content
    previous_context = " ".join(conversational_history)
    print(f"{user_question=}\n")
    sql_query = sql_chain.run({"question": user_question, "schema": db_schema,"history":previous_context}).strip()
    # sql_query = sql_chain.invoke({"question": user_question, "schema": db_schema,"history":previous_context}).strip()

    sql_query = sql_query.replace("\\_", "_")  # Fix escaped underscores
    sql_query = re.sub(r"\\([^\w])", r"\1", sql_query)  # Remove backslashes before non-word characters
    print(f"{sql_query=}\n")
    conversational_history.append(f"User: {user_question}")
    # print(sql_query,"converted_query")
    return {
        "messages": state["messages"] + [AIMessage(content=sql_query)]
    }

def Execute_query(state: State) -> State:
    query_content = state["messages"][-1].content # Clean up formatting
    query_content = re.sub(r"^```json\s*|\s*```$", "", query_content.strip(), flags=re.IGNORECASE)
    print(f"{query_content=}\n")
    try:
        # Try to parse JSON if the query is wrapped in JSON format
        query_dict = json.loads(query_content)
        print(f"{query_dict=}\n")
       
        query = query_dict.get("sql_query", "").strip()
        print(f"{query=}\n")
        
    except json.JSONDecodeError:
        # If JSON parsing fails, assume it's already a plain SQL string
        query = query_content.strip()
    

    # Ensure query is properly formatted (removing unwanted escape sequences)
    query = query.replace("\\_", "_")  # Fix escaped underscores
    query = re.sub(r"\\([^\w])", r"\1", query)  # Remove unnecessary backslashes
    print(f"{query=}\n")
    
    # print("Executing Query:", query)  # Debugging
    try:
        # Execute the SQL query and handle exceptions      
        result = sql_db_query.invoke(query)
        print(f"{result=}\n")
    except Exception as e:
        result = f"SQL Execution Error: {str(e)}"

    return {
        "messages": state["messages"] + [AIMessage(content=str(result))]
    }
def correct_sql_logic_node(state: State) -> State:
    """
    Extract values from the state and call correct_sql_logic with the proper arguments.

    Assumptions:
      - state["messages"][0] contains the original HumanMessage (question)
      - state["messages"][1] contains the AIMessage from convert_to_sql (generated SQL query)
      - state["messages"][2] contains the AIMessage from Execute_query (query output)
    Adjust the indices if your state structure is different.
    """
    # Extract the necessary values from the state
    question = state["messages"][0].content
    print(f"{question=}\n")
    query = state["messages"][1].content
    print(f"{query=}\n")


    # model4 = init_chat_model("Llama-70b-8912",model_provider="groq")
    model4 = init_chat_model("gemini-2.0-flash",model_provider = "google_genai")



    # Call your correct_sql_logic function with the extracted values
    corrected_response = correct_sql_logic(query, question, model4)
    print(f"{corrected_response=}\n")
    # Append the corrected query (or analysis) as a new message to the state
    try:
        parsed_response = json.loads(corrected_response)  # Parse response
        print(f"{parsed_response=}\n")
        corrected_sql = parsed_response.get("sql_query", "").strip()  # Extract SQL query only
        print(f"{corrected_sql=}\n")
    except json.JSONDecodeError:
        corrected_sql = corrected_response.strip()  # If not JSON, return as is
    # print("Logical",corrected_sql)
    return {"messages": state["messages"] + [AIMessage(content=corrected_sql)]}
def correct_sql_logic(query: str, question: str, model) -> str:
    verification_prompt = PromptTemplate(
        input_variables=["question", "sql_query"],
        template="""
You are an expert SQL validator. Your task is to verify whether the given SQL query correctly answers the user's question.

User Prompt:
{question}

AI-Generated SQL Query:
{sql_query}

Instructions:
- If the SQL query is in JSON format, extract only the SQL query from the "sql_query" field.
- If the SQL query is correct, return it in **exactly** the same format without any modifications.
- If the SQL query is incorrect or needs improvement, return **only** the corrected SQL query in the following JSON format, with no explanations or extra text:

{{
    "sql_query": "<UPDATED_SQL_QUERY>"
}}
"""
    )



    chain = LLMChain(llm=model, prompt=verification_prompt)
    response = chain.run({"question": question, "sql_query": query})
    print(f"{response=}\n")

    # Ensure only JSON is returned without extra formatting issues
    try:
        response_json = json.loads(response)  # Parse response to ensure it's valid JSON
        print(f"{response_json=}\n")
        return json.dumps(response_json, indent=4)  # Return formatted JSON
    except json.JSONDecodeError:
        return response.strip()  # Return raw response if not valid JSON
    
def correct_sql_syntax_node(state: State) -> State:
    # Extract the query from the state, e.g. from the last message
    query = state["messages"][1].content
    error = state["messages"][-1].content
    print(f"syntax correction {query=}\n")
    print(f"syntax correction {error=}\n")
    # global model  # or however your model is available
    model3 = init_chat_model("gemini-2.0-flash",model_provider = "google_genai")
    # Call correct_sql_syntax with the extracted query and the model
    corrected_response = correct_sql_syntax(query,error, model3)
    print(f"{corrected_response=}\n")
    return {"messages": state["messages"] + [AIMessage(content=corrected_response)]}    

def correct_sql_syntax(query: str,error:str, model) -> str:
    """
    Analyze the provided SQL query for syntax errors and logical mistakes, then output a corrected version.

    Args:
        query (str): The SQL query to analyze.
        model: An initialized LLM (e.g., using init_chat_model) that supports our prompt.

    Returns:
        str: The analysis and corrected SQL query as generated by the LLM.
    """
    # Define the prompt template that instructs the LLM as an expert in SQLite3 and SQLAlchemy.
    # prompt_template = PromptTemplate(
    #     input_variables=["query","Error"],
    #     # template="""
    #         You are an expert in SQL. Your task is to analyze the following SQL query, identify any syntax errors if need any improvement do it , and provide only a corrected version of the query in json format with no explaination or discription.
    #         don't explain any kind of correction done by you just return the corrected query in json.

    #         SQL Query:
    #         {query}

    #         Error:
    #         {Error}

    #         {{
    #       "sql_query": "<UPDATED_SQL_QUERY>"
    #         }}
    #        """
    # )
    prompt_template = PromptTemplate(
        input_variables=["query","error","schema"],
        template="""
            You are an SQLite3 expert. I will provide you with an SQL query and any error messages and schema of data.

Your task is to:
1. Identify and fix any syntax errors in the query
2. If the error is due to functions not supported in SQLite3, replace them with SQLite3-compatible alternatives
3. If no direct SQLite3 equivalent exists, create your own implementation using SQLite3's native functions
4. Ensure the query follows SQLite3 syntax and conventions
5. Columns in query should be used from schema only.
Return ONLY the corrected query in this JSON format
        {{
          "sql_query": "<UPDATED_SQL_QUERY>"
        }}
Do not include any explanations, descriptions, or commentary - just the JSON object with the corrected query.

Important: The query MUST be compatible with SQLite3 dialect specifically.

            SQL Query:
            {query}

            Error:
            {error}
            
            schema:
            {schema}

        """
    )
    chain = LLMChain(llm=model, prompt=prompt_template)

    # Run the chain with the query as input.
    response = chain.run({"query": query,"error":error,"schema":db_schema})
    print(f"{response=}\n")
    return response

def to_nlp(state: State) -> State:
  output = state["messages"][-1].content
  question = state["messages"][0].content
  for_nlp = PromptTemplate(
      input_variables=["question","output"],
      template = """
      You are an expert SQL interpreter and data analyst. I will provide you with:
1) The user's original question about data
2) The raw SQL query results

Your task is to:
- Interpret the SQL results and provide a clear, natural language explanation of what the data shows
- Connect the findings directly to the user's original question
- Highlight key insights, patterns, or notable information in the data
- Present the information in a conversational, easy-to-understand format
- Do NOT display raw results in the format [(output,)] or any similar technical format
- Format numbers appropriately (with commas for thousands, appropriate decimal places, etc.)
- Use appropriate context and business terminology when explaining the results
- If relevant, suggest possible next questions or analyses the user might want to consider

Remember that the user does not see the SQL query or raw results - they rely on your explanation to understand what the data shows.
      Question: {question}
      Output: {output}
      """
  )
  chain = LLMChain(llm=model1,prompt=for_nlp)
  response = chain.run({"question":question,"output":output})
  return {
        "messages": state["messages"] + [AIMessage(content=str(response))]
    }
def should_continue(state: State) -> str:
    """
    Determine whether the workflow should continue to the tools node or end.

    This function inspects the last message's content. If it contains the word "Error",
    it indicates that the model encountered an issue and we want to continue processing with tools.
    Otherwise, if no error is found, we return "end" to finish the workflow.

    Args:
        state (State): The current state dictionary containing the conversation/messages.

    Returns:
        str: "continue" to branch to the tools node or "end" to finish the workflow.
    """
    last_message = state["messages"][-1].content
    if "OperationalError" in last_message:
        return "Syntax_Correction"
    else:
        return "to_nlp"
# prompt = st.chat_input("Ask Your Query?")
if csv_file is not None:
    workflow =StateGraph(State)
    workflow.add_node("convert_to_sql",convert_to_sql)
    workflow.add_node("Correct_sql_logic",correct_sql_logic_node)
    workflow.add_node("Correct_sql_syntax",correct_sql_syntax_node)
    workflow.add_node("Execute_query",Execute_query)
    workflow.add_node("to_nlp", to_nlp)
    # workflow.add_edge("convert_to_sql","Execute_query")
    workflow.add_edge(START,"convert_to_sql")
    workflow.add_edge("convert_to_sql","Correct_sql_logic")
    workflow.add_edge("Correct_sql_logic","Execute_query")
    workflow.add_conditional_edges("Execute_query",
    should_continue,
    {
        "to_nlp":"to_nlp",
        "Syntax_Correction": "Correct_sql_syntax"

    })
    workflow.add_edge("Correct_sql_syntax", "Execute_query")
    workflow.add_edge("to_nlp", END)
    memory = MemorySaver()
    app=workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}
    if "messages" not in st.session_state:
       st.session_state.messages = []
    st.session_state.messages = trim_messages(st.session_state.messages)
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.write(message["content"])
    prompt = st.chat_input("Ask Your Query?")

    if prompt:
        # Append user message to session
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Initialize the workflow state
        initial_state = {
            "messages": [HumanMessage(content=prompt)]
        }

        # Invoke LangGraph workflow
        response_state = app.invoke(initial_state,config)
        print(f"{response_state=}\n")

        # Extract response
        response = response_state["messages"][-1].content if "messages" in response_state else ""
        print(f"{response=}\n")
        # Append response only if it's valid
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response_state["messages"][1].content if "messages" in response_state else "")
                st.write(response)

