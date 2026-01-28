# eR2m AI Data Science Team (An Army Of Copilots)

**An AI-powered data science team of copilots that uses agents to help you perform common data science tasks 10X faster**.

Star ⭐ This GitHub (Takes 2 seconds and means a lot).

---

The AI Data Science Team of Copilots includes Agents that specialize in promo data extraction, transformation, loading, and interpretation of various business problems like:

- Data Extraction
- Data Transformation
- Promo Loading
- Analysis-Ready Data Extraction
- Promo Performance Overview
- And more

## Data Science Agents

This project is a work in progress. New data science agents will be released soon.

![Data Science Team](img\er2m_data_science_team.png)

### Agents Available Now

1. **Data Extraction Agent:** Performs Data Extraction steps including handling missing values, and data type conversions.
2. **Data Transformation Agent:** Converts the promo planner data into raw data.

### Agents Coming Soon

1. **eR2m Supervisor:** Forms task list. Moderates sub-agents. Returns completed assignment.
2. **Data Loading Agent:** Loads promo mechanics, channels and products to MySQL database.

## Disclaimer

**This project is for testing purposes only.**

- It is not intended to replace your company's data science team
- No warranties or guarantees provided

By using this software, you agree to use it solely for testing purposes.

## Table of Contents

- [Your AI Data Science Team (An Army Of Copilots)](#your-ai-data-science-team-an-army-of-copilots)
  - [Companies That Want An AI Data Science Team Copilot](#companies-that-want-an-ai-data-science-team-copilot)
  - [Free Generative AI For Data Scientists Workshop](#free-generative-ai-for-data-scientists-workshop)
  - [Data Science Agents](#data-science-agents)
    - [Agents Available Now](#agents-available-now)
    - [Agents Coming Soon](#agents-coming-soon)
  - [Disclaimer](#disclaimer)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Example 1: Feature Engineering with the Feature Engineering Agent](#example-1-feature-engineering-with-the-feature-engineering-agent)
    - [Example 2: Cleaning Data with the Data Cleaning Agent](#example-2-cleaning-data-with-the-data-cleaning-agent)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

```bash
pip install git+https://github.com/eddychetz/eroute2market.git --upgrade
```

## Usage

### Example 1: Data Transformation with the Data Transformation Agent

```python
data_transformation_agent = make_data_transformation_agent(model = llm)

response = data_transformation_agent.invoke({
    "user_instructions": "Make sure to scale and center numeric features",
    "data_raw": df.to_dict(),
    "max_retries":3, 
    "retry_count":0
})
```

```bash
---DATA TRANSFORMATION AGENT----
    * CREATE DATA TRANSFORMER CODE
    * EXECUTING AGENT CODE
    * EXPLAIN AGENT CODE
```

### Example 2: Cleaning Data with the Data Cleaning Agent

```python
data_extraction_agent = make_data_extraction_agent(model = llm)

response = data_extraction_agent.invoke({
    "user_instructions": "Don't remove outliers when extracting the data.",
    "data_raw": df.to_dict(),
    "max_retries":3, 
    "retry_count":0
})
```

```bash
---DATA EXTRACTINO AGENT----
    * CREATE DATA EXTRACTOR CODE
    * EXECUTING AGENT CODE
    * EXPLAIN AGENT CODE
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. See LICENSE file for details.
