# coding:utf-8
import sys
import time
import numpy as np
import paddlehub as hub
import ujson
import random
from paddlehub.common.logger import logger

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
                 show_ids=False,
                 do_lower_case=True,
                 process_id=0,
                 retry=3,
                 load_balance='bind'):
        self.process_id = process_id
        self.reader_flag = False
        self.batch_size = 16
        self.embedding_size = emb_size
        self.max_seq_len = max_seq_len
        self.profile = profile
        self.model_name = model_name
        self.show_ids = show_ids
        self.do_lower_case = do_lower_case
        self.con_list = []
        self.con_index = 0
        self.load_balance = load_balance
        self.server_list = []
        self.feed_var_names = ''
        self.retry = retry

    def connect(self, server='127.0.0.1:8010'):
        self.server_list.append(server)
        '''
        con = httplib.HTTPConnection(ip, port)
        self.con_list.append(con)
        '''
    def connect_all_server(self, server_list):
        for server_str in server_list:
            self.server_list.append(server_str)
            '''
            ip, port = server_str.split(':')
            port = int(port)
            self.con_list.append(httplib.HTTPConnection(ip, port))
            '''
    def data_convert(self, text):
        if self.reader_flag == False:
            module = hub.Module(name=self.model_name)
            inputs, outputs, program = module.context(trainable=True,
                                                      max_seq_len=128)
            input_ids = inputs["input_ids"]
            position_ids = inputs["position_ids"]
            segment_ids = inputs["segment_ids"]
            input_mask = inputs["input_mask"]
            self.feed_var_names = input_ids.name + ';' + position_ids.name + ';' + segment_ids.name + ';' + input_mask.name
            self.reader = hub.reader.ClassifyReader(
                vocab_path=module.get_vocab_path(),
                dataset=None,
                max_seq_len=self.max_seq_len,
                do_lower_case=self.do_lower_case)
            self.reader_flag = True

        return self.reader.data_generator(batch_size=self.batch_size,
                                          phase='predict',
                                          data=text)

    def infer(self, request_msg):
        if self.load_balance == 'round_robin':
            try:
                cur_con = httplib.HTTPConnection(
                    self.server_list[self.con_index])
                cur_con.request('POST', "/BertService/inference", request_msg,
                                {"Content-Type": "application/json"})
                response = cur_con.getresponse()
                response_msg = response.read()
                response_msg = ujson.loads(response_msg)
                self.con_index += 1
                self.con_index = self.con_index % len(self.server_list)
                return response_msg

            except BaseException as err:
                logger.warning("Infer Error with server {} : {}".format(
                    self.server_list[self.con_index], err))
                if len(self.server_list) == 0:
                    logger.error('All server failed, process will exit')
                    return 'fail'
                else:
                    self.con_index += 1
                    return 'retry'

        elif self.load_balance == 'random':
            try:
                random.seed()
                self.con_index = random.randint(0, len(self.server_list) - 1)
                logger.info(self.con_index)
                cur_con = httplib.HTTPConnection(
                    self.server_list[self.con_index])
                cur_con.request('POST', "/BertService/inference", request_msg,
                                {"Content-Type": "application/json"})
                response = cur_con.getresponse()
                response_msg = response.read()
                response_msg = ujson.loads(response_msg)

                return response_msg
            except BaseException as err:

                logger.warning("Infer Error with server {} : {}".format(
                    self.server_list[self.con_index], err))
                if len(self.server_list) == 0:
                    logger.error('All server failed, process will exit')
                    return 'fail'
                else:
                    self.con_index = random.randint(0,
                                                    len(self.server_list) - 1)
                    return 'retry'

        elif self.load_balance == 'bind':

            try:
                self.con_index = int(self.process_id) % len(self.server_list)
                cur_con = httplib.HTTPConnection(
                    self.server_list[self.con_index])
                cur_con.request('POST', "/BertService/inference", request_msg,
                                {"Content-Type": "application/json"})
                response = cur_con.getresponse()
                response_msg = response.read()
                response_msg = ujson.loads(response_msg)

                return response_msg
            except BaseException as err:

                logger.warning("Infer Error with server {} : {}".format(
                    self.server_list[self.con_index], err))
                if len(self.server_list) == 0:
                    logger.error('All server failed, process will exit')
                    return 'fail'
                else:
                    self.con_index = int(self.process_id) % len(
                        self.server_list)
                    return 'retry'

    def encode(self, text):
        if type(text) != list:
            raise TypeError('Only support list')
        #start = time.time()
        self.batch_size = len(text)
        data_generator = self.data_convert(text)
        start = time.time()
        request_time = 0
        result = []
        for run_step, batch in enumerate(data_generator(), start=1):
            request = []
            copy_start = time.time()
            token_list = batch[0][0].reshape(-1).tolist()
            pos_list = batch[0][1].reshape(-1).tolist()
            sent_list = batch[0][2].reshape(-1).tolist()
            mask_list = batch[0][3].reshape(-1).tolist()
            for si in range(self.batch_size):
                instance_dict = {}
                instance_dict["token_ids"] = token_list[si *
                                                        self.max_seq_len:(si +
                                                                          1) *
                                                        self.max_seq_len]
                instance_dict["sentence_type_ids"] = sent_list[si *
                                                               self.max_seq_len:
                                                               (si + 1) *
                                                               self.max_seq_len]
                instance_dict["position_ids"] = pos_list[si *
                                                         self.max_seq_len:(si +
                                                                           1) *
                                                         self.max_seq_len]
                instance_dict["input_masks"] = mask_list[si *
                                                         self.max_seq_len:(si +
                                                                           1) *
                                                         self.max_seq_len]
                request.append(instance_dict)

            copy_time = time.time() - copy_start
            #request
            request = {"instances": request}
            request["max_seq_len"] = self.max_seq_len
            request["emb_size"] = self.embedding_size
            request["feed_var_names"] = self.feed_var_names
            request_msg = ujson.dumps(request)
            if self.show_ids:
                logger.info(request_msg)
            request_start = time.time()
            response_msg = self.infer(request_msg)
            retry = 0
            while type(response_msg) == str and response_msg == 'retry':
                if retry < self.retry:
                    retry += 1
                    logger.info('Try to connect another servers')
                    response_msg = self.infer(request_msg)
                else:
                    logger.error('Infer failed after {} times retry'.format(
                        self.retry))
                    break
            retry = 0
            for msg in response_msg["instances"]:
                for sample in msg["instances"]:
                    result.append(sample["values"])

            #request end
            request_time += time.time() - request_start
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
        '''
        for con in self.con_list:
            con.close()
        '''


def test():

    bc = BertService(model_name='bert_chinese_L-12_H-768_A-12',
                     emb_size=768,
                     show_ids=True,
                     do_lower_case=True)
    bc.connect('127.0.0.1:8010')
    result = bc.encode([
        ["远上寒山石径斜"],
    ])
    #print(result[0])
    bc.close()


if __name__ == '__main__':
    test()
