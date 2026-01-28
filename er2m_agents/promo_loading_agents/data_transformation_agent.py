# EROUTE2MARKET  AGENTS
# ER2M DATA SCIENCE TEAM
# ***
# * Agents: Feature Transformation Agent

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import interrupt, Command
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

# * Feature Engineering Agent

def make_data_tranformation_agent(model, log=False, log_path=None, human_in_the_loop=False):
    """
    Creates a data transformation agent that can be run on a dataset. The agent applies various data transformation
    techniques, such as encoding categorical variables, scaling numeric variables, creating interaction terms,
    and generating polynomial features. The agent takes in a dataset and user instructions and outputs a Python
    function for data transformation. It also logs the code generated and any errors that occur.

    The agent is instructed to apply the following data transformation techniques:

    - Read excel sheet with current and upcoming promotions to get promo mechanics - get_mechanics()
    - Get promo products description ready for loading - get_products()
    - Return a two data frames containing the transformed mechanics and products.
    - Any specific instructions provided by the user

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to "logs/".
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the feature engineering instructions. Defaults to False.

    Examples
    -------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from er2m_agents.promo_loading_agents.data_transformation_agent import make_data_tranformation_agent

    llm = ChatOpenAI(model="gpt-4o-mini")

    feature_engineering_agent = make_data_tranformation_agent(llm)

    df = pd.read_csv("https://github.com/eddychetz/eroute2market/raw/refs/heads//data/PFM Promotional Planner Current and Upcoming 2026.xlsx")

    response = data_transformation_agent.invoke({
        "user_instructions": None,
        "data_raw": df.to_dict(),
        "max_retries": 3,
        "retry_count": 0
    })

    pd.DataFrame(response['data_transformed'])
    ```

    Returns
    -------
    app : langchain.graphs.StateGraph
        The data transformation agent as a state graph.
    """
    llm = model

    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = "logs/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        data_transformer_function: str
        data_transformer_error: str
        data_transformed: dict
        max_retries: int
        retry_count: int

    def recommend_data_transformation_steps(state: GraphState):
        """
        Recommend a series of data transformation steps based on the input data.
        These recommended steps will be appended to the user_instructions.
        """
        print("---DATA TRANSFORMATION AGENT----")
        print("    * RECOMMEND DATA TRANSFORMATION STEPS")

        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Data Transformation Expert. Given the following information about the data,
            recommend a series of numbered steps to take to transform the data into raw data.
            The steps should be tailored to the data characteristics and should be helpful
            for a data transformation agent that will be implemented.

            General Steps:
            Things that should be considered in the data transformation steps:

            * Read excel sheet with current and upcoming promotions to get promo mechanics - get_mechanics()
            * Get promo products description ready for loading - get_products()
            * Return a two data frames containing the transformed mechanics and products.
            * Any specific instructions provided by the user


            Custom Steps:
            * Analyze the data to determine if any additional data transformation steps are needed.
            * Recommend steps that are specific to the data provided. Include why these steps are necessary or beneficial.
            * If no additional steps are needed, simply state that no additional steps are required.

            IMPORTANT:
            Make sure to take into account any additional user instructions that may add, remove or modify some of these steps. Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested. Include comments if something is done because a user requested.

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
            input_variables=["user_instructions", "data_head", "data_description", "data_info"]
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
            "data_description": df.describe(include='all').to_string(),
            "data_info": info_text
        })

        return {"recommended_steps": "\n\n# Recommended Steps:\n" + recommended_steps.content.strip()}

    # Human Review
    def human_review(state: GraphState) -> Command[Literal["recommend_data_transformation_steps", "create_data_transformation_code"]]:
        return node_func_human_review(
            state=state,
            prompt_text="Is the following data transformation instructions correct? (Answer 'yes' or provide modifications)\n{steps}",
            yes_goto="create_feature_engineering_code",
            no_goto="recommend_data_transformation_steps",
            user_instructions_key="user_instructions",
            recommended_steps_key="recommended_steps"
        )

    # def human_review(state: GraphState) -> Command[Literal["recommend_data_transformation_steps", "create_data_transformation_code"]]:
    #     print("    * HUMAN REVIEW")

    #     user_input = interrupt(
    #         value=f"Is the following data transformation instructions correct? (Answer 'yes' or provide modifications to make to make them correct)\n{state.get('recommended_steps')}",
    #     )

    #     if user_input.strip().lower() == "yes":
    #         goto = "create_data_transformation_code"
    #         update = {}
    #     else:
    #         goto = "recommend_data_transformation_steps"
    #         modifications = "Modifications: \n" + user_input
    #         if state.get("user_instructions") is None:
    #             update = {
    #                 "user_instructions": modifications,
    #             }
    #         else:
    #             update = {
    #                 "user_instructions": state.get("user_instructions") + modifications,
    #             }

    #     return Command(goto=goto, update=update)

    def create_data_transformation_code(state: GraphState):
        print("    * CREATE DATA TRANSFORMATION CODE")

        data_transformation_prompt = PromptTemplate(
            template="""

            You are a Data Transformation Agent. Your job is to create a data_transformer() function that can be run on the data provided using the following recommended steps.

            Recommended Steps:
            {recommended_steps}

            Use this information about the data to help determine how to data transform for the data:

            Target Variable: {target_variable}

            Sample Data (first 100 rows):
            {data_head}

            Data Description:
            {data_description}

            Data Info:
            {data_info}

            You can use Pandas, Numpy, and Scikit Learn libraries to data transform the data.

            Return Python code in ```python``` format with a single function definition, data_transformer(data_raw), including all imports inside the function.

            Return code to provide the data transformation function:

            def data_transformer(data_raw):
                import pandas as pd
                import numpy as np
                ...
                return data_transformed

            Best Practices and Error Preventions:
            - Handle missing values in numeric and date features before transformations.


            """,
            input_variables=["recommeded_steps", "target_variable", "data_head", "data_description", "data_info"]
        )

        data_transformation_agent = data_transformation_prompt | llm | PythonOutputParser()

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()

        response = data_transformation_agent.invoke({
            "recommended_steps": state.get("recommended_steps"),
            "target_variable": state.get("target_variable"),
            "data_head": df.head().to_string(),
            "data_description": df.describe(include='all').to_string(),
            "data_info": info_text
        })

        # Relocate imports inside the function
        response = relocate_imports_inside_function(response)

        # For logging: store the code generated
        if log:
            with open(log_path + 'data_transformer.py', 'w') as file:
                file.write(response)

        return {"data_transformer_function": response}

    # Execute Data Transformation Code
    def execute_data_transformation_code(state):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_transformed",
            error_key="data_transformer_error",
            code_snippet_key="data_transformer_function",
            agent_function_name="data_transformer",
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            post_processing=lambda df: df.to_dict(),
            error_message_prefix="An error occurred during data transformation: "
        )

    # Fix Data Transformation Code
    def fix_data_transformation_code(state: GraphState):
        data_transformer_prompt = """
        You are a Data Transformation Agent. Your job is to fix the data_transformer() function that currently contains errors.

        Provide only the corrected function definition.

        Broken code:
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_transformer_function",
            error_key="data_transformer_error",
            llm=llm,
            prompt_template=data_transformer_prompt,
            log=log,
            log_path=log_path,
            log_file_name="data_transformer.py"
        )

    # Explain Data Transformation Code
    def explain_data_transformation_code(state: GraphState):
        return node_func_explain_agent_code(
            state=state,
            code_snippet_key="data_transformer_function",
            result_key="messages",
            error_key="data_transformer_error",
            llm=llm,
            role="data_transformation_agent",
            explanation_prompt_template="""
            Explain the data transformation steps performed by this function. Keep the explanation clear and concise.\n\n# Data Transformation Agent:\n\n{code}
            """,
            success_prefix="# Data Transformation Agent:\n\n ",
            error_message="The Data Transformation Agent encountered an error during data transformation. Data could not be explained."
        )

    # Create the graph
    node_functions = {
        "recommend_data_transformation_steps": recommend_data_transformation_steps,
        "human_review": human_review,
        "create_data_transformation_code": create_data_transformation_code,
        "execute_data_transformation_code": execute_data_transformation_code,
        "fix_data_transformation_code": fix_data_transformation_code,
        "explain_data_transformation_code": explain_data_transformation_code
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_data_transformation_steps",
        create_code_node_name="create_data_transformation_code",
        execute_code_node_name="execute_data_transformation_code",
        fix_code_node_name="fix_data_transformation_code",
        explain_code_node_name="explain_data_transformation_code",
        error_key="data_transformer_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=MemorySaver() if human_in_the_loop else None
    )

    return app