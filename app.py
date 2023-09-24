import time
import sys
import streamlit as st
import string
import os
from io import StringIO 
import pdb
import json
import torch
import requests
import socket
from streamlit_image_select import image_select





use_case = {"1":"Image background removal - (upload any picture and remove background)","2":"Masking foreground for downstream inpainting task"}
mask_types = {
"rgba - makes background white":"rgba",
"green - makes the background green":"green",
"blur - blurs background":"blur",
"map - makes the foreground white and rest black ":"map"
}



APP_NAME = "twc/salient_object_detection"
INFO_URL = "https://www.taskswithcode.com/stats/"
TMP_DIR="tmp_dir"
TMP_SEED = 1



        

def get_views(action):
    ret_val = 0
    #return "{:,}".format(ret_val)
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    if ("view_count" not in st.session_state):
        try:
           app_info = {'name': APP_NAME,"action":action,"host":hostname,"ip":ip_address}
           res = requests.post(INFO_URL, json = app_info).json()
           print(res)
           data = res["count"]
        except Exception as e:
           data = 0
           print(f"Exception in get_views - uncached case. view count not cached: {str(e)}")
        ret_val = data
        st.session_state["view_count"] = data
    else:
        ret_val = st.session_state["view_count"]
        if (action != "init"):
           try:
               app_info = {'name': APP_NAME,"action":action,"host":hostname,"ip":ip_address}
               print(app_info)
               res = requests.post(INFO_URL, json = app_info)
               print(res)
               res = res.json()
           except Exception as e:
                print(f"Exception in get_views - Non init case. view count not cached: {str(e)}")
    return "{:,}".format(ret_val)
        



def construct_model_info_for_display(model_names,api_info):
    options_arr  = []
    #markdown_str = f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\"><br/><b>Models evaluated ({len(model_names)})</b><br/></div>"
    markdown_str = f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\"><br/><b>Model evaluated </b><br/></div>"
    markdown_str += f"<div style=\"font-size:2px; color: #2f2f2f; text-align: left\"><br/></div>"
    for node in model_names:
        options_arr .append(node["name"])
        if (node["mark"] == "True"):
            markdown_str += f"<div style=\"font-size:16px; color: #5f5f5f; text-align: left\">&nbsp;•&nbsp;Model:&nbsp;<a href=\'{node['paper_url']}\' target='_blank'>{node['name']}</a><br/>&nbsp;&nbsp;&nbsp;&nbsp;Code released by:&nbsp;<a href=\'{node['orig_author_url']}\' target='_blank'>{node['orig_author']}</a><br/>&nbsp;&nbsp;&nbsp;&nbsp;Model info:&nbsp;<a href=\'{node['sota_info']['sota_link']}\' target='_blank'>{node['sota_info']['task']}</a></div>"
            if ("Note" in node):
                markdown_str += f"<div style=\"font-size:16px; color: #a91212; text-align: left\">&nbsp;&nbsp;&nbsp;&nbsp;{node['Note']}<a href=\'{node['alt_url']}\' target='_blank'>link</a></div>"
            markdown_str += "<div style=\"font-size:16px; color: #5f5f5f; text-align: left\"><br/></div>"

    
    markdown_str += f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\"><b>{api_info['desc']}</b><br/></div>"
    for method in api_info["methods"]:
        lang = method["lang"]
        example = open(method["usage"]).read()
        markdown_str += f"<div style=\"font-size:16px; color: #5f5f5f; text-align: center\"><b>{lang} usage</b></div>"
        markdown_str += f"<div style=\"font-size:14px; color: #bfbfbf; text-align: left\">{example}<br/></div>"
        
    markdown_str += "<div style=\"font-size:12px; color: #9f9f9f; text-align: left\"><b><br/>Note:</b><br/>•&nbsp;Uploaded files are loaded into non-persistent memory for the duration of the computation. They are not cached</div>"
    markdown_str += "<div style=\"font-size:12px; color: #9f9f9f; text-align: left\"><br/><a href=\'https://github.com/taskswithcode/salient_object_detection_app.git\' target='_blank'>Github code</a> for this app</div>"

    return options_arr,markdown_str


def init_page():
    st.set_page_config(page_title='TasksWithCode', page_icon="logo.png", layout='centered', initial_sidebar_state='auto',
            menu_items={
             'About': 'This app was created by taskswithcode. http://taskswithcode.com'
             
              })
    col,pad = st.columns([85,15])

    with col:
        st.image("long_form_logo_with_icon.png")


def run_test(config,input_file_name,display_area,uploaded_file,mask_type):
    global TMP_SEED
    display_area.text("Processing request...")
    try:
        if (uploaded_file is None):
            file_data = open(input_file_name, "rb")
            r = requests.post(config["SERVER_ADDRESS"], data={"mask":mask_type}, files={"test":file_data})
        else:
            file_data = uploaded_file.read()
            file_name = f"{TMP_DIR}/{TMP_SEED}_{str(time.time()).replace('.','_')}_{uploaded_file.name}"
            TMP_SEED += 1
            with open(file_name,"wb") as fp:
                fp.write(file_data)
            file_data = open(file_name, "rb")
            r = requests.post(config["SERVER_ADDRESS"], data={"mask":mask_type}, files={"test":file_data})
            os.remove(file_name)
        print("Servers response:",r.status_code,len(r.content))
        if (r.status_code == 200):
            size = "{:,}".format(len(r.content))
            return {"response":r.content,"size":size}
        else:
            return {"error":f"API request failed {r.status_code}"}
    except Exception as e:
        st.error("Some error occurred during prediction" + str(e))
        #st.stop()
        return {"error":f"Exception in performing image masking: {str(e)}"}
    return {} 


    

def display_results(results,response_info,mask):
    main_sent = f"<div style=\"font-size:14px; color: #2f2f2f; text-align: left\">{response_info}<br/><br/></div>"
    body_sent = []
    download_data = {}
    main_sent = main_sent + "\n" + '\n'.join(body_sent)
    st.markdown(main_sent,unsafe_allow_html=True)
    st.image(results["response"], caption=f'Output of Image background removal with mask: {mask}')
    st.session_state["download_ready"]  = results["response"]


def init_session():
    init_page()
    st.session_state["model_name"] = "insprynet"
    st.session_state["download_ready"] = None    
    st.session_state["model_name"] = "ss_test"
    st.session_state["file_name"] = "default"
    st.session_state["mask_type"] = "rgba"
 
def app_main(app_mode,example_files,model_name_files,api_info_files,config_file):
  init_session()
  with open(example_files) as fp:
        example_file_names = json.load(fp) 
  with open(model_name_files) as fp:
        model_names = json.load(fp)
  with open(config_file) as fp:
        config = json.load(fp)
  with open(api_info_files) as fp:
        api_info = json.load(fp)
  curr_use_case = use_case[app_mode].split(".")[0]
  curr_use_case = use_case[app_mode].split(".")[0]
  st.markdown("<h5 style='text-align: center;'>Image foreground masking or background removal</h5>", unsafe_allow_html=True)
  st.markdown(f"<div style='color: #4f4f4f; text-align: left'>Image masking using state-of-the-art models for salient object detection(SOD). SOD use cases are<br/>&nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;{use_case['1']}<br/>&nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;{use_case['2']}</div>", unsafe_allow_html=True)
  st.markdown(f"<div style='color: #9f9f9f; text-align: right'>views:&nbsp;{get_views('init')}</div>", unsafe_allow_html=True)


  try:
      
      
      with st.form('twc_form'):

        step1_line = "Upload an image or choose an example image below"
        uploaded_file = st.file_uploader(step1_line, type=["png","jpg","jpeg"])

        selected_file_name = image_select("Select image", ["twc_samples/sample1.jpg", "twc_samples/sample2.jpg", "twc_samples/sample3.jpg", "twc_samples/sample4.jpg"])


        st.write("")
        mask_type = st.selectbox(label=f'Select type of masking',  
                    options = list(dict.keys(mask_types)), index=0,  key = "twc_mask_types")
        mask_type = mask_types[mask_type]
        st.write("")
        submit_button = st.form_submit_button('Run')
        options_arr,markdown_str = construct_model_info_for_display(model_names,api_info)

        
        input_status_area = st.empty()
        display_area = st.empty()
        if submit_button:
            start = time.time()
            if uploaded_file is not None:
                st.session_state["file_name"]  = uploaded_file.name
            else:
                st.session_state["file_name"]  = selected_file_name
            st.session_state["mask_type"]  = mask_type
            display_area.empty()
            results = run_test(config,st.session_state["file_name"],display_area,uploaded_file,mask_type)
            with display_area.container():
                if ("error" in results):
                    st.error(results["error"])
                else:
                    device = 'GPU' if torch.cuda.is_available() else 'CPU'
                    response_info = f"Computation time on {device}: {time.time() - start:.2f} secs for image size: {results['size']} bytes"
                    display_results(results,response_info,mask_type)
                    #st.json(results)
            get_views("submit")
      st.download_button(
         label="Download results as png",
         data= st.session_state["download_ready"] if st.session_state["download_ready"] != None else "",
         disabled = False if st.session_state["download_ready"] != None else True,
         file_name= (st.session_state["model_name"] + "_"  + st.session_state["mask_type"] + "_" +  '_'.join(st.session_state["file_name"].split(".")[:-1]) + ".png").replace("/","_"),
         mime='image/png',
         key ="download" 
        )
      
      

  except Exception as e:
    st.error("Some error occurred during loading" + str(e))
    #st.stop()  
	
  st.markdown(markdown_str, unsafe_allow_html=True)
  
 

if __name__ == "__main__":
   app_main("1","sod_app_examples.json","sod_app_models.json","sod_apis.json","config.json")

