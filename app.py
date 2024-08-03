import streamlit as st, pandas as pd, os, io
from modeci_mdf.mdf import Model, Graph, Node, Parameter, OutputPort
from modeci_mdf.utils import load_mdf_json, load_mdf, load_mdf_yaml
from modeci_mdf.execution_engine import EvaluableGraph, EvaluableOutput
import json
import requests
st.set_page_config(layout="wide", page_icon="logo.png", page_title="Model Description Format", menu_items={
        'Report a bug': "https://github.com/ModECI/MDF/",
        'About': "ModECI (Model Exchange and Convergence Initiative) is a multi-investigator collaboration that aims to develop a standardized format for exchanging computational models across diverse software platforms and domains of scientific research and technology development, with a particular focus on neuroscience, Machine Learning and Artificial Intelligence. Refer to https://modeci.org/ for more."
    })
header1, header2 = st.columns([1,12], vertical_alignment="top")
with header1:
    st.image("logo.png", width=100)
with header2:
    st.title("Welcome to Model Description Format")
st.write("Lets get started! Choose one of the following methods.")
# models: Purpose: To store the state of the model and update the model
def run_simulation(param_inputs, mdf_model):
    mod_graph = mdf_model.graphs[0]
    nodes = mod_graph.nodes
    for node in nodes:
        parameters = node.parameters
        outputs = node.output_ports
        eg = EvaluableGraph(mod_graph, verbose=False)
        # eo = EvaluableOutput(mod_graph, verbose=)
        duration = param_inputs["Simulation Duration (s)"]
        dt = param_inputs["Time Step (s)"]
        print("hi me param_inputs inside run simulation", param_inputs)
        t = 0
        times = []
        output_values = {op.value: [] for op in outputs}
        while t <= duration:
            times.append(t)
            if t == 0:
                eg.evaluate()
            else:
                eg.evaluate(time_increment=dt)

            for param in output_values:
                if any(operator in param  for operator in "+-/*"):
                    eval_param = eg.enodes[node.id].evaluable_outputs[param]
                    
                else:
                    eval_param = eg.enodes[node.id].evaluable_parameters[param] 
                output_values[param].append(eval_param.curr_value)
            t += dt
        chart_data = pd.DataFrame(output_values)
        chart_data['Time'] = times
        chart_data.set_index('Time', inplace=True)
        show_simulation_results(chart_data)
    return None
# views: Purpose: To display the state of the model 
def show_simulation_results(chart_data):
    st.line_chart(chart_data, use_container_width=True, height=400)

def show_mdf_graph(mdf_model):
    st.subheader("MDF Graph")
    mdf_model.to_graph_image(engine="dot", output_format="png", view_on_render=False, level=3, filename_root=mdf_model.id, only_warn_on_fail=(os.name == "nt"))
    image_path = mdf_model.id + ".png"
    st.sidebar.image(image_path, caption="Model Graph Visualization")

def show_json_output(mdf_model):
    st.subheader("JSON Output")
    st.json(mdf_model.to_json())

def view_tabs(mdf_model, param_inputs): # view
    tab1, tab2, tab3 = st.tabs(["Simulation Results", "MDF Graph", "Json Output"])
    with tab1:
        # mod_graph = mdf_model.graphs[0]
        # nodes = mod_graph.nodes
        # for node in nodes:
        #     st.write("Node Name: ", node.id)
        #     node_parameters = {}
        #     for param in node.parameters:
        #         node_parameters[param.id] = node_parameters[param.id] = param.value if param.value else param.time_derivative
        #     st.write("Node Parameters: ", node_parameters)   
        run_simulation(param_inputs, mdf_model) # model
        #     show_simulation_results(chart_data) # view
    with tab2:
        show_mdf_graph(mdf_model) # view
    with tab3:
        show_json_output(mdf_model) # view

def parameter_form_to_update_model_and_view(mdf_model, parameters, param_inputs, mod_graph, nodes):
    form = st.form(key="parameter_form")
    valid_inputs = True
    for node_wise_parameter_key, node_wise_parameter in enumerate(parameters):
        # st.write("hi i am b", node_wise_parameter)
        for i, param in enumerate(node_wise_parameter):
            if isinstance(param.value, str) or param.value is None:
                continue  
            key = f"{param.id}_{i}"
            if mdf_model.metadata:
                value = form.text_input(f"{param.metadata.get('description', param.id)} ({param.id})", value=str(param.value), key=key)
            else:
                value = form.text_input(f"{param.id}", value=str(param.value), key=key)
            
            try:
                param_inputs[param.id] = float(value)
            except ValueError:
                st.error(f"Invalid input for {param.id}. Please enter a valid number.")
                valid_inputs = False

    sim_duration = form.text_input("Simulation Duration (s)", value=str(param_inputs["Simulation Duration (s)"]), key="sim_duration")
    time_step = form.text_input("Time Step (s)", value=str(param_inputs["Time Step (s)"]), key="time_step")

    try:
        param_inputs["Simulation Duration (s)"] = float(sim_duration)
    except ValueError:
        st.error("Invalid input for Simulation Duration. Please enter a valid number.")
        valid_inputs = False
    try:
        param_inputs["Time Step (s)"] = float(time_step)
    except ValueError:
        st.error("Invalid input for Time Step. Please enter a valid number.")
        valid_inputs = False

    run_button = form.form_submit_button("Run Simulation")
    if run_button:
        if valid_inputs:
            for b in parameters:
                for param in b:
                    if param.id in param_inputs:
                        param.value = param_inputs[param.id]
            view_tabs(mdf_model, param_inputs)
        # else:
        #     st.error("Please correct the invalid inputs before running the simulation.")

# def upload_file_and_load_to_model():
#     st.write("Choose how to load the model:")
#     load_option = st.radio("", ("Upload File", "GitHub URL", "Example Models"), )
#     st.write("Choose how to load the model:")
#     if load_option == "Upload File":
#         uploaded_file = st.file_uploader("Choose a JSON/YAML/BSON file", type=["json", "yaml", "bson"])
#         if uploaded_file is not None:
#             file_content = uploaded_file.getvalue()
#             file_extension = uploaded_file.name.split('.')[-1].lower()
#             return load_model_from_content(file_content, file_extension)

#     elif load_option == "GitHub URL":
#         st.write("sample_github_url = https://raw.githubusercontent.com/ModECI/MDF/development/examples/MDF/NewtonCoolingModel.json")
#         github_url = st.text_input("Enter GitHub raw file URL:", placeholder="Enter GitHub raw file URL")
#         if github_url:
#             try:
#                 response = requests.get(github_url)
#                 response.raise_for_status()
#                 file_content = response.content
#                 file_extension = github_url.split('.')[-1].lower()
#                 return load_model_from_content(file_content, file_extension)
#             except requests.RequestException as e:
#                 st.error(f"Error loading file from GitHub: {e}")
#                 return None

#     elif load_option == "Example Models":
#         example_models = {
#             "Newton Cooling Model": "https://raw.githubusercontent.com/ModECI/MDF/development/examples/MDF/NewtonCoolingModel.json",
#             "ABCD": "https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/ABCD.json",
#             "FN": "https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/FN.mdf.json",
#             "States": "https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/States.json",
#             "Other Model 4": "https://example.com/other_model_4.json"
#         }

#         selected_model = st.selectbox("Choose an example model", list(example_models.keys()))
#         if selected_model:
#             example_url = example_models[selected_model]
#             try:
#                 response = requests.get(example_url)
#                 response.raise_for_status()
#                 file_content = response.content
#                 file_extension = example_url.split('.')[-1].lower()
#                 return load_model_from_content(file_content, file_extension)
#             except requests.RequestException as e:
#                 st.error(f"Error loading example model: {e}")
#                 return None

#     st.write("Try out example files:")
#     return None

def upload_file_and_load_to_model():
    # col1, col2 = st.columns(2)
    # with col1:
    uploaded_file = st.file_uploader("Choose a JSON/YAML/BSON file", type=["json", "yaml", "bson"])
    if uploaded_file is not None:
        file_content = uploaded_file.getvalue()
        file_extension = uploaded_file.name.split('.')[-1].lower()
        return load_model_from_content(file_content, file_extension)
    github_url = st.text_input("Enter GitHub raw file URL:", placeholder="Enter GitHub raw file URL")
    if github_url:
        try:
            response = requests.get(github_url)
            response.raise_for_status()
            file_content = response.content
            file_extension = github_url.split('.')[-1].lower()
            return load_model_from_content(file_content, file_extension)
        except requests.RequestException as e:
            st.error(f"Error loading file from GitHub: {e}")
            return None
    # with col2:
    example_models = {
        "Newton Cooling Model": "https://raw.githubusercontent.com/ModECI/MDF/development/examples/MDF/NewtonCoolingModel.json",
        "ABCD": "https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/ABCD.json",
        "FN": "https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/FN.mdf.json",
        "States": "https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/States.json",
        "Other Model 4": "https://example.com/other_model_4.json"
    }
    selected_model = st.selectbox("Choose an example model", list(example_models.keys()), index=None)
    if selected_model:
        example_url = example_models[selected_model]
        try:
            response = requests.get(example_url)
            response.raise_for_status()
            file_content = response.content
            file_extension = example_url.split('.')[-1].lower()
            return load_model_from_content(file_content, file_extension)
        except requests.RequestException as e:
            st.error(f"Error loading example model: {e}")
            return None
    # st.button("Newton Cooling Model", on_click=load_mdf_json(""))
    return None


def load_model_from_content(file_content, file_extension):
    try:
        if file_extension == 'json':
            json_data = json.loads(file_content)
            mdf_model = Model.from_dict(json_data)
        elif file_extension in ['yaml', 'yml']:
            mdf_model = load_mdf_yaml(io.BytesIO(file_content))
        else:
            st.error("Unsupported file format. Please use JSON or YAML files.")
            return None
        
        st.session_state.original_mdf_model = mdf_model  # Save the original model
        st.session_state.mdf_model_yaml = mdf_model  # Save the current model state
        return mdf_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    mdf_model = upload_file_and_load_to_model() # controller
    if mdf_model:
        mod_graph = mdf_model.graphs[0]
        nodes = mod_graph.nodes
        parameters = []
        for node in nodes:
            parameters.append(node.parameters)
        param_inputs = {}
        if mdf_model.metadata:
            preferred_duration = float(mdf_model.metadata.get("preferred_duration", 10))
            preferred_dt = float(mdf_model.metadata.get("preferred_dt", 0.1))
        else:
            preferred_duration = 100
            preferred_dt = 0.1
        param_inputs["Simulation Duration (s)"] = preferred_duration
        param_inputs["Time Step (s)"] = preferred_dt
        parameter_form_to_update_model_and_view(mdf_model, parameters, param_inputs, mod_graph, nodes)

if __name__ == "__main__":
    main()



