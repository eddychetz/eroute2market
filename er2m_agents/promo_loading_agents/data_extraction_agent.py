# EROUTE2MARKET  AGENTS
# E2RM DATA SCIENCE TEAM
# ***
# * Agents: BevCo Promo Loading Agent

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal, Optional, Tuple
import operator

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

import os
import io
import pandas as pd

from er2m_agents.templates.agent_templates import(
    node_func_execute_agent_code_on_data,
    node_func_human_review,
    node_func_fix_agent_code,
    node_func_explain_agent_code,
    create_coding_agent_graph
)
from er2m_agents.tools.parsers import PythonOutputParser
from er2m_agents.tools.regex import relocate_imports_inside_function

# Setup

LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Agent: BevCo Promo Loading Agent

def make_data_extraction_agent(
    model, log=False,
    log_path=None,
    human_in_the_loop=False
    ) -> Tuple[pd.DataFrame, str]:
    """
    Creates a BevCo Promo Extraction Agent. The agent can be used to extract promo data from a CSV file.

    The step-by-step logic is as follows:

    1. Read the CSV file and load the data into a pandas DataFrame.
    2. Check if the file is imported correctly using a function.
    3. If the file is imported correctly, return the DataFrame.
    4. If the file is not imported correctly, ask the user to fix the file.
    5. Oncethe file is fixed, repeat steps 1-4 until the file is imported correctly.

    Parameters
    ----------

    model : langchain.llms.base.LLM
        The language model to use to generate code.

    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.

    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to
        "logs/".

    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the data cleaning instructions. Defaults to False.

    Examples
    --------

    ```python
    import pandas as pd
    from er2m_agent.promo_loading_agents import data_extraction_agent

    df = pd.read_csv("https://github.com/eddychetz/eroute2market/raw/refs/heads//data/PFM Promotional Planner Current and Upcoming 2026.xlsx")

    response = extraction_agent.make_data_extraction_agent(
        model=your_llm_model,
        log=True,
        human_in_the_loop=False
    )

    pd.DataFrame(response['promo_loaded'])
    ```

    Returns:
    -------
    app : langgrapg.graphs.StateGraph - A BevCo Promo Extraction Agent instance.
    The promo extraction agent as a state graph
    """

    # Assign the model to llm

    llm = model

    # Setup Log Directory

    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        data_extractor_function: str
        data_extractor_error: str
        data_extracted: dict
        max_retries: int
        retry_count: int

    #  Recommended Promo Loading Steps
    def recommend_extraction_steps(state: GraphState):
        """
        Recommend a series of data cleaning steps based on the input data.
        These recommended steps will be appended to the user_instructions.
        """

        print("---PROMO EXTRACTION AGENT----")
        print("    * RECOMMEND PROMO EXTRACTION STEPS")

        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a data engineer expert. Based on the following user instructions and the preview of
            the promo data, recommend a series of data extraction steps that should be taken before transforming
            the data into raw data. Provide the steps in a numbered list format. The steps should
            be tailored to data characteristics and should helpful for the bevco promotions extraction
            agent that will be implemented.

            General Steps to consider::
            * Get file directory from user using a function.
            * Read the data using a function.
            * Validate the data types of each column using a function.
                * Check for missing values and handle them appropriately - logging promos without start and/or end date.

            Custom Steps:
            * Analyze the data to determine any specific cleaning steps that are necessary based on the data characteristics.
            * Recommen steps that are specific to the data provided.
            * If no specific steps are needed, state that no additional steps are required.

            IMPORTANT:
            Make sure to take into account any additional user instructions that may add, remove or modify some of the steps.
            Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested.
            Include comments if somethingis done because a user requested.

            User instructions:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Data Sample (first 50 rows):
            {data_head}

            Data Description:
            {data_description}

            Data Info:
            {data_info}

            Return the steps as a bullet point list (no code, just the steps).

            """,
            input_variables=["user_instructions", "data_head","data_description","data_info"]
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "data_head": df.head(50).to_string(),
            "data_description": df.describe().to_string(),
            "data_info": info_text
        })

        return {"recommended_steps": "\n\n# Recommended Steps:\n" + recommended_steps.content.strip()}

    # Create Data Extraction Code
    def create_data_extractor_code(state: GraphState):
        print("    * CREATE DATA EXTRACTOR CODE")

        data_extraction_prompt = PromptTemplate(
            template="""
            You are a Promo Extraction Agent. Your job is to create a data_extractor() function that can be run on the data provided using the following recommended steps.

            Recommended Steps:
            {recommended_steps}

            You can use Pandas, and Numpy libraries to extract the data.

            Use this information about the data to help determine how to extract the data:

            Sample Data (first 100 rows):
            {data_head}

            Data Description:
            {data_description}

            Data Info:
            {data_info}

            Return Python code in ```python ``` format with a single function definition, data_extractor(data_raw), that incldues all imports inside the function. 

            Return code to provide the data extraction function:

            def data_extract(data_raw):
                import pandas as pd
                import numpy as np
                ...
                return data_extracted

            Best Practices and Error Preventions:

            Always ensure that when assigning the output of fit_transform() from SimpleImputer to a Pandas DataFrame column, you call .ravel() or flatten the array, because fit_transform() returns a 2D array while a DataFrame column is 1D.

            """,
            input_variables=["recommended_steps", "data_head", "data_description", "data_info"]
        )

        data_extraction_agent = data_extraction_prompt | llm | PythonOutputParser()

        data_raw = state.get("data_raw")

        df = pd.DataFrame.from_dict(data_raw)


        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()

        response = data_extraction_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "data_head": df.head().to_string(),
            "data_description": df.describe().to_string(),
            "data_info": info_text
        })

        response = relocate_imports_inside_function(response)

        # For logging: store the code generated:
        if log:
            with open(log_path + 'data_extractor.py', 'w') as file:
                file.write(response)

        return {"data_extractor_function" : response}

    # Human Review of Recommended Steps
    def human_review(state: GraphState) -> Command[Literal["recommend_extraction_steps", "create_data_extractor_code"]]:
        return node_func_human_review(
            state=state,
            prompt_text="Is the following data extraction instructions correct? (Answer 'yes' or provide modifications)\n{steps}",
            yes_goto="create_data_extractor_code",
            no_goto="recommend_extraction_steps",
            user_instructions_key="user_instructions",
            recommended_steps_key="recommended_steps"
        )

    # Execute Data Extraction Code
    def execute_data_extractor_code(state):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_extracted",
            error_key="data_extractor_error",
            code_snippet_key="data_extractor_function",
            agent_function_name="data_extractor",
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            post_processing=lambda df: df.to_dict(),
            error_message_prefix="An error occurred during data extraction: "
        )

    # Fix Data Extraction Code
    def fix_data_extractor_code(state: GraphState):
            data_extractor_prompt = """
            You are a Data Extraction Agent. Your job is to create a data_extractor() function that can be run on the data provided. The function is currently broken and needs to be fixed.

            Make sure to only return the function definition for data_extractor().

            Return Python code in ```python``` format with a single function definition, data_extractor(data_raw), that includes all imports inside the function.

            This is the broken code (please fix):
            {code_snippet}

            Last Known Error:
            {error}
            """

            return node_func_fix_agent_code(
                state=state,
                code_snippet_key="data_extractor_function",
                error_key="data_extraction_error",
                llm=llm,
                prompt_template=data_extractor_prompt,
                log=log,
                log_path=log_path,
                log_file_name="data_extractor.py"
            )

    # Explain Data Extraction Code
    def explain_data_extractor_code(state: GraphState):
        return node_func_explain_agent_code(
            state=state,
            code_snippet_key="data_extractor_function",
            result_key="messages",
            error_key="data_extraction_error",
            llm=llm,
            role="data_extraction_agent",
            explanation_prompt_template="""
            Explain the data extraction steps that the data extraction agent performed in this function.
            Keep the summary succinct and to the point.\n\n# Data Extraction Agent:\n\n{code}
            """,
            success_prefix="# Data Extraction Agent:\n\n ",
            error_message="The Data Extraction Agent encountered an error during data extraction. Data could not be explained."
        )

    # Define the graph
    node_functions = {
        "recommend_extraction_steps": recommend_extraction_steps,
        "human_review": human_review,
        "create_data_extractor_code": create_data_extractor_code,
        "execute_data_extractor_code": execute_data_extractor_code,
        "fix_data_extractor_code": fix_data_extractor_code,
        "explain_data_extractor_code": explain_data_extractor_code
    }

    # Create the graph app
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_extraction_steps",
        create_code_node_name="create_data_extractor_code",
        execute_code_node_name="execute_data_extractor_code",
        fix_code_node_name="fix_data_extractor_code",
        explain_code_node_name="explain_data_extractor_code",
        error_key="data_extractor_error",
        human_in_the_loop=human_in_the_loop,  # or False
        human_review_node_name="human_review",
        checkpointer=MemorySaver() if human_in_the_loop else None
    )

    return app

        # * Check if the file is imported correctly using a function.

        # * Extract the data using a function.
        # * Validate the data types of each column.
        # * Check for missing values and handle them appropriately - logging promos without start and/or end date.
        # * Ensure that date formats are consistent and correct.
        # * Get promo mechanics ready for loading.
        # * Load the promo mechanics and channels into config_promo and config_promo_channel tables respectively.
        # * Get promo products description ready for loading.
        # * Load the promo products ids into config_promo_product table.
        # * Get summary report for loaded promos.
        # * Get summary report for promos that failed to load.
        # * Get summary report for errors encountered during loading.

# * Ensure that date formats are consistent and correct.
#             * Get promo mechanics ready for loading.
#             * Load the promo mechanics and channels into config_promo and config_promo_channel tables respectively.
#             * Get promo products description ready for loading.
#             * Load the promo products ids into config_promo_product table.
#             * Get summary report for loaded promos.
#             * Get summary report for promos that failed to load.
#             * Get summary report for errors encountered during loading.
