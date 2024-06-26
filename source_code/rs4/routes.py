from rs4 import app
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from rs4.functions import *
from rs4.forms import *
import time
sub_dir_path = None


def count_bin_files(directory):
    """
    Recursively counts the total number of .bin files in a directory.
    
    Args:
    - directory (str): Path to the directory.
    
    Returns:
    - total_count (int): Total number of .bin files.
    """
    total_count = 0
    
    # Iterate over all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # If it's a directory, recursively call the function
        if os.path.isdir(item_path):
            total_count += count_bin_files(item_path)
        # If it's a file and ends with .bin, increment the count
        elif os.path.isfile(item_path) and item.endswith('.bin'):
            total_count += 1
    
    return total_count

def create_subdirectory(directory_name, subdirectory_name):
    # Create the main directory if it doesn't exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    
    # Create the subdirectory within the main directory
    subdirectory_path = os.path.join(directory_name, subdirectory_name)
    os.makedirs(subdirectory_path, exist_ok=True)
    
    return subdirectory_path

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/features', methods=['GET', 'POST'])
def upload_file():
    global sub_dir_path
    blocks_accesed = 0
    if request.method == 'POST':
        # Save the uploaded file in the current directory
        file = request.files['fileToUpload']
        graph_type = request.form['graphType']
        if(graph_type == "unweighted"):
            graph_type = 1
        else: 
            graph_type = 2
        dir_name = '/home/sukhomay/Desktop/DBMS_Lab/P/graphs'
        sub_dir_name = file.filename.split('.')[0]
        sub_dir_path = create_subdirectory(dir_name, sub_dir_name)
        file.save(os.path.join(sub_dir_path, secure_filename(file.filename)))
        
        st_time = time.time()
        create_directories(sub_dir_path,sub_dir_name,graph_type)
        directory_name = dir_name + "/" + sub_dir_name
        make_node_info(sub_dir_path)
        make_adj_list(sub_dir_path)
        make_File_Index(sub_dir_path)
        tarjan_scc(sub_dir_path)
        wcc(sub_dir_path)
        calculate_pagerank(sub_dir_path)
        calculate_rank(sub_dir_path)
        count_cycles(sub_dir_path)
        end_time = time.time()
        diff_time = end_time - st_time
        diff_time = int(1000*diff_time)
        blocks = get_block()
        with open(f"{sub_dir_path}/Meta_data.txt", 'a') as meta_file:
            meta_file.write(f"Time_taken : {diff_time}\n")
            meta_file.write(f"numfiles : {count_bin_files(sub_dir_path)}\n")
    
    return render_template("features.html")


@app.route('/metadata', methods=['GET', 'POST'])
def metadata_page():
    global sub_dir_path

    metadata = get_Meta_data(sub_dir_path)
    return render_template('metadata.html', metadata = metadata)

@app.route('/indegree', methods=['GET', 'POST'])
def indegree_page():
    global sub_dir_path

    form=Nodeform()
    indeg=None
    diff_time = 0
    blocks_accesed = 0
    if form.validate_on_submit():
        node=form.node.data
        st_time = time.time()
        indeg=get_in_degree(sub_dir_path,node)
        en_time = time.time()
        diff_time = float(1000*(en_time-st_time))
        blocks_accesed=get_block()
        redirect(url_for('indegree_page'))
    return render_template('indegree.html',form=form,indeg=indeg, blocks=blocks_accesed+1, time=diff_time)

@app.route('/outdegree', methods=['GET', 'POST'])
def outdegree_page():
    global sub_dir_path
    global blocks_accesed
    form=Nodeform()
    outdeg=None
    diff_time = 0
    blocks_accesed = 0
    if form.validate_on_submit():
        node=form.node.data
        
        st_time = time.time()
        outdeg=get_out_degree(sub_dir_path,node)
        en_time = time.time()
        diff_time = float(1000*(en_time-st_time))
        blocks_accesed=get_block()

        redirect(url_for('outdegree_page'))
    return render_template('outdegree.html',form=form,outdeg=outdeg, blocks=blocks_accesed+1, time=diff_time)

@app.route('/rank', methods=['GET', 'POST'])
def rank_page():
    global sub_dir_path
    form=Nodeform()
    rank=None
    pgrank=None
    diff_time = 0
    blocks_accesed = 0
    if form.validate_on_submit():
        node=form.node.data
        
        st_time = time.time()
        rank=get_rank(sub_dir_path,node)
        pgrank=get_page_rank(sub_dir_path,node)
        en_time = time.time()
        diff_time = float(1000*(en_time-st_time))
        blocks_accesed=get_block()
        pgrank/=1000000000
        redirect(url_for('rank_page'))
    return render_template('rank.html',form=form,rank=rank,pgrank=pgrank, blocks=blocks_accesed+1, time=diff_time)


@app.route('/knn', methods=['GET', 'POST'])
def knn_page():
    global sub_dir_path
    form=Twoinputform()
    sorted_tuples=None
    diff_time = 0
    blocks_accesed = 0
    if form.validate_on_submit():
        node=form.inp1.data
        k=form.inp2.data
        
        st_time = time.time()
        d=KNN(sub_dir_path,node,k)
        sorted_tuples = [(value,key) for key, value in d.items()]
        sorted_tuples=sorted(sorted_tuples)
        en_time = time.time()
        blocks_accesed=get_block()
        diff_time = float(1000*(en_time-st_time))

        redirect(url_for('knn_page'))
    return render_template('knn.html',form=form,knn=sorted_tuples, blocks=blocks_accesed+1, time=diff_time)

@app.route('/shortest_distance', methods=['GET', 'POST'])
def shortest_distance_page():
    global sub_dir_path
    form=Twoinputform()
    dist=None
    diff_time = 0
    blocks_accesed = 0
    if form.validate_on_submit():
        print("hello***************************")
        node1=form.inp1.data
        node2=form.inp2.data

        print(node1, node2)
        
        st_time = time.time()
        dist=dijkstra(sub_dir_path,node1,node2)
        if(dist==float('inf')):
            dist = -1
        en_time = time.time()
        diff_time = float(1000*(en_time-st_time))
        blocks_accesed=get_block()
        redirect(url_for('shortest_distance_page'))
    return render_template('shortest_distance.html',form=form,dist=dist, blocks=blocks_accesed+1, time=diff_time)

@app.route('/rank_list', methods=['GET', 'POST'])
def rank_list_page():
    global sub_dir_path
    form=Twoinputform()
    ranklist = None
    diff_time = 0
    blocks_accesed = 0
    if form.validate_on_submit():
        l=form.inp1.data
        r=form.inp2.data
        
        st_time = time.time()
        ranknodes = get_ranklist(sub_dir_path, l, r)
        ranklist = []
        for node in ranknodes:
            ranklist.append((l, node))
            l+=1
        en_time = time.time()
        diff_time = float(1000*(en_time-st_time))
        blocks_accesed=get_block()
        redirect(url_for('rank_list_page'))
    return render_template('rank_list.html',form=form,ranklist=ranklist, blocks=blocks_accesed+1, time=diff_time)


@app.route('/component', methods=['GET', 'POST'])
def component_page():
    global sub_dir_path
    form=Twoinputform()
    have_same_WCC = None
    have_same_SCC = None
    diff_time = 0
    blocks_accesed = 0
    if form.validate_on_submit():
        node1=form.inp1.data
        node2=form.inp2.data
        
        st_time = time.time()
        have_same_SCC = checkInSameSCC(sub_dir_path, node1, node2)
        have_same_WCC = checkInSameWCC(sub_dir_path, node1, node2)
        en_time = time.time()
        diff_time = float(1000*(en_time-st_time))
        blocks_accesed=get_block()
        redirect(url_for('component_page'))
    return render_template('component.html',form=form, have_same_SCC=have_same_SCC, have_same_WCC=have_same_WCC, blocks=blocks_accesed+1, time=diff_time)






