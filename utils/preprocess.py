import argparse
import os
import numpy as np
import sys
sys.path.append('/home/wenxiaofei.wxf/crshust')
from adacrs.utils.utils_load_save import save_dataset, load_dataset, save_kg
from adacrs.utils.utils_load_save import DATA_DIR, TMP_DIR
from adacrs.utils.utils_load_save import LAST_FM, LAST_FM_STAR, YELP, YELP_STAR, INS
from adacrs.data.processors.lastfm_data_process import LastFmDataset
from adacrs.data.processors.lastfm_star_data_process import LastFmStarDataset
from adacrs.data.processors.yelp_data_process import YelpDataset
from adacrs.data.processors.lastfm_graph import LastFmGraph
from adacrs.data.processors.yelp_graph import YelpGraph
from adacrs.data.processors.insurance_data_process import INSDataset
from adacrs.data.processors.insurance_graph import INSGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        type=str,
        default=INS,
        choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR, INS],
        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR, INS}.')
    args = parser.parse_args()
    DatasetDict = {
        LAST_FM: LastFmDataset,
        LAST_FM_STAR: LastFmStarDataset,
        YELP: YelpDataset,
        YELP_STAR: YelpDataset,
        INS: INSDataset,
    }
    GraphDict = {
        LAST_FM: LastFmGraph,
        LAST_FM_STAR: LastFmGraph,
        YELP: YelpGraph,
        YELP_STAR: YelpGraph,
        INS: INSGraph,
    }
    
    
    # Create 'data_name' instance for data_name.
    print('Load', args.data_name, 'from file...')
    print(TMP_DIR[args.data_name])
    
    # make sure for TMP_DIR
    if not os.path.isdir(TMP_DIR[args.data_name]):
        os.makedirs(TMP_DIR[args.data_name])
    
    # Dataset.__init__
    dataset = DatasetDict[args.data_name](DATA_DIR[args.data_name])
    
    # dump dataset.pkl to TMP_DIR
    save_dataset(args.data_name, dataset) 
    print('Save', args.data_name, 'dataset successfully!')

    # Generate graph instance for 'data_name'
    print('Create', args.data_name, 'graph from data_name...')
    
    # load dataset object .pkl
    dataset = load_dataset(args.data_name)
    
    # generate kg.pkl and dump it to TMP_DIR
    kg = GraphDict[args.data_name](dataset)
    save_kg(args.data_name, kg)
    print('Save', args.data_name, 'graph successfully!')


def construct(kg):
    users = kg.G['user'].keys()
    items = kg.G['item'].keys()
    features = kg.G['feature'].keys()
    num_node = len(users) + len(items) + len(features)
    graph = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            if i < len(users) and j < len(users) + len(items):
                graph[i][j] = 1
                graph[j][i] = 1
            elif i >= len(users) and i < len(users) + len(items):
                if j - len(users) - len(items) in kg.G['item'][
                        i - len(users)]['belong_to']:
                    graph[i][j] = 1
                    graph[j][i] = 1
            else:
                pass
    print(graph)
    return graph


if __name__ == '__main__':
    main()
