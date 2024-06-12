from __future__ import division
import os
import sys
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'

class Spice:
    """
    Main Class to compute the SPICE metric 
    """

    def float_convert(self, obj):
        try:
          return float(obj)
        except:
          return np.nan
      
    def fetch_tuples(self, tuples):
        result_tuples = []
        for item in tuples:
            result_tuples.append(item['tuple'])
        return result_tuples
    
    def find_common(self, tuple_A, tuple_B):
        common = 0
        for item in tuple_A:
            if item in tuple_B:
                common += 1

        return common
    
    def get_identity_tuples(self, data):
        person_ids = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11"]
        filtered_tuples = [item for item in data if any(person_id in item for person_id in person_ids)]
        action_tuples = [tup for tup in filtered_tuples if len(tup) > 1]
        id_tuples = list(set([tuple(tup) for tup in filtered_tuples if len(tup) == 1]))
        id_tuples = [list(tup) for tup in id_tuples]
        return action_tuples, id_tuples

    def calculate_metrics(self, pred_tuples, ref_tuples):
        common = self.find_common(pred_tuples, ref_tuples)
        total_pred = len(pred_tuples)
        total_ref = len(ref_tuples)
        if total_pred == 0 or total_ref == 0:
          return 0

        precision = common / total_pred
        recall = common / total_ref

        if precision + recall == 0:
           return 0
        
        f1_score = (2 * precision * recall)/(precision + recall)
        return f1_score
            
         

      
    
    
    def compute_score(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())
        
        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
              "image_id" : id,
              "test" : hypo[0],
              "refs" : ref
            })

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir=os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
          os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir,
                                              mode='w+')
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        cache_dir=os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
          os.makedirs(cache_dir)
        spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
          '-cache', cache_dir,
          '-out', out_file.name,
          '-detailed',
          '-silent'
        ]
        subprocess.check_call(spice_cmd, 
            cwd=os.path.dirname(os.path.abspath(__file__)))

        # Read and process results
        with open(out_file.name) as data_file:    
          results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)
        

        imgId_to_scores = {}
        spice_scores = []
        ispice_scores = []
        for item in results:
          imgId_to_scores[item['image_id']] = item['scores']
          spice_scores.append(self.float_convert(item['scores']['All']['f']))
          pred_tuples = self.fetch_tuples(item['test_tuples'])
          ref_tuples = self.fetch_tuples(item['ref_tuples'])
          ia_pred_tuples, id_pred_tuples = self.get_identity_tuples(pred_tuples)
          ia_ref_tuples, id_ref_tuples = self.get_identity_tuples(ref_tuples)

          if(len(ia_pred_tuples) != 0):
            ia_spice_score = self.calculate_metrics(ia_pred_tuples, ia_ref_tuples)
            id_spice_score = self.calculate_metrics(id_pred_tuples, id_ref_tuples)
            ispice_scores.append(ia_spice_score * id_spice_score)
          
        average_spice_score = np.mean(np.array(spice_scores))
        average_ispice_score = np.mean(np.array(ispice_scores))

        return average_spice_score, spice_scores, average_ispice_score, ispice_scores

    def method(self):
        return "iSPICE"
    
