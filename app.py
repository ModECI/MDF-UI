import streamlit as st, pandas as pd, os, io
from modeci_mdf.mdf import Model, Graph, Node, Parameter, OutputPort
from modeci_mdf.utils import load_mdf_json, load_mdf, load_mdf_yaml
from modeci_mdf.execution_engine import EvaluableGraph, EvaluableOutput
import json
import numpy as np
import requests
st.set_page_config(layout="wide", page_icon="page_icon.png", page_title="Model Description Format", menu_items={
        'Report a bug': "https://github.com/ModECI/MDF/",
        'About': "ModECI (Model Exchange and Convergence Initiative) is a multi-investigator collaboration that aims to develop a standardized format for exchanging computational models across diverse software platforms and domains of scientific research and technology development, with a particular focus on neuroscience, Machine Learning and Artificial Intelligence. Refer to https://modeci.org/ for more."
    })

def reset_simulation_state():
    """Reset simulation-related session state variables."""
    if 'simulation_results' in st.session_state:
        del st.session_state.simulation_results
    if 'selected_columns' in st.session_state:
        del st.session_state.selected_columns

def run_simulation(param_inputs, mdf_model, stateful):
    mod_graph = mdf_model.graphs[0]
    nodes = mod_graph.nodes
    all_node_results = {}
    if stateful:
        duration = param_inputs["Simulation Duration (s)"]
        dt = param_inputs["Time Step (s)"]
    
        
        
        for node in nodes:
            eg = EvaluableGraph(mod_graph, verbose=False)
            t = 0
            times = []
            node_outputs = {op.value : [] for op in node.output_ports}
            node_outputs['Time'] = []
            
            while t <= duration:
                times.append(t)
                if t == 0:
                    eg.evaluate()
                else:
                    eg.evaluate(time_increment=dt)

                node_outputs['Time'].append(t)
                for op in node.output_ports:
                    eval_param = eg.enodes[node.id].evaluable_outputs[op.id]
                    output_value = eval_param.curr_value
                    if isinstance(output_value, (list, np.ndarray)):
                        scalar_value = output_value[0] if len(output_value) > 0 else np.nan
                        node_outputs[op.value].append(float(scalar_value))
                    else:
                        node_outputs[op.value].append(float(output_value))
                t += dt
            
            all_node_results[node.id] = pd.DataFrame(node_outputs).set_index('Time')
        
        return all_node_results
    else:
        for node in nodes:
            eg = EvaluableGraph(mod_graph, verbose=False)
            eg.evaluate()
            all_node_results[node.id] = pd.DataFrame({op.value: [float(eg.enodes[node.id].evaluable_outputs[op.id].curr_value)] for op in node.output_ports})
            
    return all_node_results
def show_simulation_results(all_node_results, stateful_nodes):
    if all_node_results is not None:
        for node_id, chart_data in all_node_results.items():
            st.subheader(f"Results for Node: {node_id}")
            if node_id in stateful_nodes:
                if 'selected_columns' not in st.session_state:
                    st.session_state.selected_columns = {node_id: {col: True for col in chart_data.columns}}
                elif node_id not in st.session_state.selected_columns:
                    st.session_state.selected_columns[node_id] = {col: True for col in chart_data.columns}
                
                # Filter the data based on selected checkboxes
                filtered_data = chart_data[[col for col, selected in st.session_state.selected_columns[node_id].items() if selected]]
                # Display the line chart with filtered data
                st.line_chart(filtered_data, use_container_width=True, height=400)
                columns = chart_data.columns
                checks = st.columns(8)
                if len(columns) > 0 and len(st.session_state.selected_columns[node_id])>1:
                    for l, column in enumerate(columns):
                        with checks[l]:
                            st.checkbox(
                                f"{column}",
                                value=st.session_state.selected_columns[node_id][column],
                                key=f"checkbox_{node_id}_{column}",
                                on_change=update_selected_columns,
                                args=(node_id, column,)
                            )
                #show checkboxes horizontally
                # in case we late go back to vertical
                # for column in columns:
                #     st.checkbox(
                #         f"{column}",
                #         value=st.session_state.selected_columns[node_id][column],
                #         key=f"checkbox_{node_id}_{column}",
                #         on_change=update_selected_columns,
                #         args=(node_id, column,)
                #     )


                
            else:
                st.write(all_node_results[node_id])

def update_selected_columns(node_id, column):
    st.session_state.selected_columns[node_id][column] = st.session_state[f"checkbox_{node_id}_{column}"]

def show_mdf_graph(mdf_model):
    st.subheader("MDF Graph")
    mdf_model.to_graph_image(engine="dot", output_format="png", view_on_render=False, level=3, filename_root=mdf_model.id, only_warn_on_fail=(os.name == "nt"))
    image_path = mdf_model.id + ".png"
    st.image(image_path, caption="Model Graph Visualization")

def show_json_model(mdf_model):
    st.subheader("JSON Model")
    st.json(mdf_model.to_json())

# st.cache_data()
def view_tabs(mdf_model, param_inputs, stateful): # view
    tab1, tab2, tab3 = st.tabs(["Simulation Results", "MDF Graph", "Json Model"])
    with tab1:
        if 'simulation_run' not in st.session_state or not st.session_state.simulation_run:
            st.write("Run the simulation to see results.")
        elif st.session_state.simulation_results is not None:
            show_simulation_results(st.session_state.simulation_results, stateful)
        else:
            st.write("No simulation results available.")

    with tab2:
        show_mdf_graph(mdf_model) # view
    with tab3:
        show_json_model(mdf_model) # view

def display_and_edit_array(array, key):
    if isinstance(array, list):
        array = np.array(array)
    
    rows, cols = array.shape if array.ndim > 1 else (1, len(array))
    
    edited_array = []
    for i in range(rows):
        row = []
        for j in range(cols):
            value = array[i][j] if array.ndim > 1 else array[i]
            edited_value = st.text_input(f"[{i}][{j}]", value=str(value), key=f"{key}_{i}_{j}")
            try:
                row.append(float(edited_value))
            except ValueError:
                st.error(f"Invalid input for [{i}][{j}]. Please enter a valid number.")
        edited_array.append(row)
    
    return np.array(edited_array)

def parameter_form_to_update_model_and_view(mdf_model):
    mod_graph = mdf_model.graphs[0]
    nodes = mod_graph.nodes
    parameters = []
    stateful_nodes = []
    stateful = False

    for node in nodes:
        for param in node.parameters:
            if param.is_stateful():
                stateful_nodes.append(node.id)
                stateful = True
                break
            else:
                stateful = False

    param_inputs = {}
    if stateful:
        if mdf_model.metadata:
            preferred_duration = float(mdf_model.metadata.get("preferred_duration", 10))
            preferred_dt = float(mdf_model.metadata.get("preferred_dt", 0.1))
        else:
            preferred_duration = 100
            preferred_dt = 0.1
        param_inputs["Simulation Duration (s)"] = preferred_duration
        param_inputs["Time Step (s)"] = preferred_dt

    with st.form(key="parameter_form"):
        valid_inputs = True
        st.write("Model Parameters:")

        for node_index, node in enumerate(nodes):
            with st.container(border=True):
                st.write(f"Node: {node.id}")
                
                # Create four columns for each node
                col1, col2, col3, col4 = st.columns(4)
                
                for i, param in enumerate(node.parameters):
                    if isinstance(param.value, str) or param.value is None:
                        continue  
                    key = f"{param.id}_{node_index}_{i}"
                    
                    # Alternate between columns
                    current_col = [col1, col2, col3, col4][i % 4]
                    
                    with current_col:
                        if isinstance(param.value, (list, np.ndarray)):
                            st.write(f"{param.id}:")
                            value = display_and_edit_array(param.value, key)
                        else:
                            if param.metadata:
                                value = st.text_input(f"{param.metadata.get('description', param.id)} ({param.id})", value=str(param.value), key=key)
                            else:
                                value = st.text_input(f"{param.id}", value=str(param.value), key=key)
                            try:
                                param_inputs[param.id] = float(value)
                            except ValueError:
                                st.error(f"Invalid input for {param.id}. Please enter a valid number.")
                                valid_inputs = False
                    
                    param_inputs[param.id] = value
        if stateful:
            st.write("Simulation Parameters:")
            with st.container(border=True):
                # Add Simulation Duration and Time Step inputs
                col1, col2 = st.columns(2)
                with col1:
                    sim_duration = st.text_input("Simulation Duration (s)", value=str(param_inputs["Simulation Duration (s)"]), key="sim_duration")
                with col2:
                    time_step = st.text_input("Time Step (s)", value=str(param_inputs["Time Step (s)"]), key="time_step")
                
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

        run_button = st.form_submit_button("Run Simulation")
        
    if run_button:
        if valid_inputs:
            for node in nodes:
                for param in node.parameters:
                    if param.id in param_inputs:
                        param.value = param_inputs[param.id]
            st.session_state.simulation_results = run_simulation(param_inputs, mdf_model, stateful)
            st.session_state.simulation_run = True
        else:
            st.error("Please correct the invalid inputs before running the simulation.")

    view_tabs(mdf_model, param_inputs, stateful_nodes)

def upload_file_and_load_to_model():
   
    uploaded_file = st.sidebar.file_uploader("Choose a JSON/YAML/BSON file", type=["json", "yaml", "bson"])
    github_url = st.sidebar.text_input("Enter GitHub raw file URL:", placeholder="Enter GitHub raw file URL")
    example_models = {
        "Newton Cooling Model": "./examples/NewtonCoolingModel.json",
        "ABCD": "./examples/ABCD.json",
        "FN": "./examples/FN.mdf.json",
        "States": "./examples/States.json",
        "Switched RLC Circuit": "./examples/switched_rlc_circuit.json",
        "Simple":"./examples/Simple.json",
        # "Arrays":"./examples/Arrays.json",
        # "RNN":"./examples/RNNs.json", # some issue
        "IAF":"./examples/IAFs.json",
        "Izhikevich Test":"./examples/IzhikevichTest.mdf.json"
    }
    selected_model = st.sidebar.selectbox("Choose an example model", list(example_models.keys()), index=None, placeholder="Dont have an MDF Model? Try some sample examples here!")
    
    if uploaded_file is not None:
        file_content = uploaded_file.getvalue()
        file_extension = uploaded_file.name.split('.')[-1].lower()
        return load_model_from_content(file_content, file_extension)

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

    
    if selected_model:
        return load_mdf_json(example_models[selected_model])



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
    if "checkbox" not in st.session_state:
        st.session_state.checkbox = False
    
    
    mdf_model = upload_file_and_load_to_model() # controller

    if mdf_model:
        st.session_state.current_model = mdf_model
        header1, header2 = st.columns([1, 8], vertical_alignment="top")
        with header1:
            with st.container():
                st.image("logo.jpg")
        with header2:
            with st.container():
                st.title("MDF: "+ mdf_model.id)
        
        parameter_form_to_update_model_and_view(mdf_model)
    else:
        header1, header2 = st.columns([1, 8], vertical_alignment="top")
        with header1:
            with st.container():
                st.image("logo.jpg")
        with header2:
            with st.container():
                st.title("Welcome to the Model Description Format UI")
        st.write("ModECI (Model Exchange and Convergence Initiative) is a multi-investigator collaboration that aims to develop a standardized format for exchanging computational models across diverse software platforms and domains of scientific research and technology development, with a particular focus on neuroscience, Machine Learning and Artificial Intelligence. Refer to https://modeci.org/ for more.")
        st.header("Let's get started! Choose one of the options on the left to load an MDF model.")
if __name__ == "__main__":
    main()



