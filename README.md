# PersonalityAdaptiveChatbot

This repository serves as an archive for the implementation of my Master's thesis, focusing on the creation of a chatbot that adapts its Large Language Model-, and transfer learning-based responses according to the Big Five personality traits of users. The project involves a variety of experiments and implementations with the Rasa framework, the Haystack and LangChain frameworks, and the Plotly Dash dashboard framework, among others. The repository is structured into various folders, each containing different components of the project. For a more detailed description of each folder, please see below.

## Repository Structure

### `actions` 

This folder contains Rasa custom actions, including an API endpoint for communication with the NLP Pipeline. It also houses the implementation of the NEO-FFI-30 test, a well-known personality assessment tool.

### `conversation_runs` 

Here, you'll find documented test runs from automated chatbot-chatbot conversations. These conversations were part of the evaluation process for the chatbot's response adaptation capabilities.

### `more_ditched_experiments` 

This directory stores discarded experiments involving Rasa for custom pipelining and the Panel dashboard framework. While these experiments did not make it into the final project, they may provide valuable insights and potential starting points for future work.

### `notebooks` 

This directory is central to the project, containing various Jupyter notebooks that document the experimental process of implementing the chatbot's core functionalities. Each sub-folder covers a different aspect of the project:

- `big5_detection`: Experiments related to Big Five personality trait modeling and exploration of the developed models.
- `haystack`: Experiments focusing on building a deployment-ready NLP pipeline with the Haystack framework.
- `langchain`: Experiments related to the adaptive Natural Language Generation (NLG) components using the LangChain framework. This includes Language Model Likelihood (LLM) evaluation, prompt engineering, and conversation memory.
- `model_persisting`: Work focused on optimizing and persisting the finalized Big Five modeling using MLFlow.
- `results`: Documentation of the results of the project, including classification scores and response generation.

### `pipeline` 

The `pipeline` folder contains the main implementation and deployment components of the project. Each sub-folder includes a Dockerfile and a Kubernetes configuration file (`k8s-*.yaml`) for reproducing the module. Here are the key sub-folders:

- `main`: Contains the custom NLP implementation, most notably the `ChatbotPipeline.py` file, which houses the deployed NLP pipeline.
- `mlflow`: Hosts the MLflow server, along with the database and artifacts needed for reproduction.
- `pipelinedashboard`: Contains the evaluation dashboard implementation, most notably the `app.py` file, which serves the main application.

### `website` 

This folder houses the static HTML page that serves the chatbot via JavaScript.

## Contributing

As this repository serves as an archive for a Master's thesis, contributing guidelines are not provided. However, anyone interested in further development or research in the field of personality-adaptive chatbots may find this repository to be a valuable resource.

## License

The project is currently not under any specific license. For any queries related to licensing or usage, please reach out directly. 
