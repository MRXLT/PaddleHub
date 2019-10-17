# coding:utf-8
import sys
import time
import numpy as np
import paddlehub as hub
import json

_ver = sys.version_info
is_py2 = (_ver[0] == 2)
is_py3 = (_ver[0] == 3)

if is_py2:
    import httplib
if is_py3:
    import http.client as httplib


class BertService():
    def __init__(self,
                 profile=False,
                 max_seq_len=128,
                 model_name="bert_uncased_L-12_H-768_A-12",
                 emb_size=768,
                 show_ids=False):
        self.reader_flag = False
        self.batch_size = 1
        self.embedding_size = emb_size
        self.max_seq_len = max_seq_len
        self.profile = profile
        self.model_name = model_name
        self.show_ids = show_ids

    def connect(self, ip='127.0.0.1', port=8010):
        self.con = httplib.HTTPConnection(ip, port)

    def data_convert(self, text):
        if self.reader_flag == False:
            module = hub.Module(name=self.model_name)
            dataset = hub.dataset.ChnSentiCorp()
            self.reader = hub.reader.ClassifyReader(
                dataset=dataset,
                vocab_path=module.get_vocab_path(),
                max_seq_len=self.max_seq_len)
            self.reader_flag = True

        return self.reader.data_generator(
            batch_size=self.batch_size, phase='predict', data=text)

    def encode(self, text):
        if type(text) != list:
            raise TypeError('Only support list')
        #start = time.time()
        self.batch_size = len(text)
        data_generator = self.data_convert(text)
        result = []
        start = time.time()
        request = []
        for run_step, batch in enumerate(data_generator(), start=1):
            copy_start = time.time()
            token_list = batch[0][0].reshape(-1).tolist()
            pos_list = batch[0][1].reshape(-1).tolist()
            sent_list = batch[0][2].reshape(-1).tolist()
            mask_list = batch[0][3].reshape(-1).tolist()
            for si in range(self.batch_size):
                instance_dict = {}
                instance_dict["token_ids"] = token_list[si * self.max_seq_len:(
                    si + 1) * self.max_seq_len]
                instance_dict["sentence_type_ids"] = sent_list[
                    si * self.max_seq_len:(si + 1) * self.max_seq_len]
                instance_dict["position_ids"] = pos_list[si * self.max_seq_len:(
                    si + 1) * self.max_seq_len]
                instance_dict["input_masks"] = mask_list[si * self.max_seq_len:(
                    si + 1) * self.max_seq_len]
                instance_dict["max_seq_len"] = self.max_seq_len
                instance_dict["emb_size"] = self.embedding_size
                request.append(instance_dict)
            copy_time = time.time() - copy_start
        request = {"instances": request}
        request_msg = json.dumps(request)
        if self.show_ids:
            print(request_msg)
        request_start = time.time()
        try:
            self.con.request('POST', "/BertService/inference", request_msg,
                             {"Content-Type": "application/json"})
            response = self.con.getresponse()
            response_msg = response.read()
            #print(response_msg)
            response_msg = json.loads(response_msg)
            for msg in response_msg["instances"]:
                for sample in msg["instances"]:
                    result.append(sample["values"])

        except httplib.HTTPException as e:
            print(e.reason)
        request_time = time.time() - request_start
        total_time = time.time() - start
        start = time.time()
        if self.profile:
            return [
                total_time, request_time, response_msg['op_time'],
                response_msg['infer_time'], copy_time
            ]
        else:
            return result

    def close(self):
        self.con.close()


def test():

    bc = BertService(
        model_name='bert_chinese_L-12_H-768_A-12', emb_size=768, show_ids=True)
    bc.connect('127.0.0.1', 8010)
    result = bc.encode([["hello"], ])
    print(len(result[0]))
    bc.close()


if __name__ == '__main__':
    test()
