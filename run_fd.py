import json
import os
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import tqdm
from sklearn.metrics import classification_report, precision_score, accuracy_score, recall_score, f1_score, \
    confusion_matrix


def df_to_json_idx_size(df, idx, size, d_columns=['Number', 'date', '序号'], need_diff=True, addi='序号'):
    """
    Convert the specified DataFrame data starting from index idx with a continuous length of size into a JSON string,
    with optional additional information showing value changes (increase/decrease) relative to previous values.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        idx (int): Starting index.
        size (int): Length of continuous data to extract.
        d_columns (list): Column names to exclude, defaults to ['Number', 'date', '序号'].
        need_diff (bool): Whether to calculate relative changes from previous values, defaults to True.

    Returns:
        str: JSON string of the selected portion.
    """
    # Get the column names of the DataFrame
    col_name_list = list(df.columns)

    addi_list = []
    if addi in col_name_list:
        addi_list = list(df.loc[idx:idx + size - 1, col_name_list][addi])

    # Remove specified column names
    for c_n in d_columns:
        try:
            col_name_list.remove(c_n)
        except Exception as e_:
            pass
            # print(f'Error removing column name: {e_}')

    # Extract data starting from idx with length size
    selected_data = df.loc[idx:idx + size - 1, col_name_list]

    if need_diff:
        # Calculate relative changes from previous values
        for col in col_name_list:
            # Ignore non-numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                selected_data[col + '_change'] = selected_data[col].diff()  # Calculate numerical difference
                selected_data[col + '_change'] = selected_data[col + '_change'].fillna(0).apply(
                    lambda x: f"(Increase {x:.2f})" if x > 0 else f"(Decrease {abs(x):.2f})" if x < 0 else "")  # Handle increase/decrease only

                # Combine value and change information
                selected_data[col] = selected_data.apply(
                    lambda row: f"{row[col]} {row[col + '_change']}".strip(), axis=1)  # Remove trailing space

        # Remove temporary change columns
        selected_data = selected_data.drop(columns=[col + '_change' for col in col_name_list if pd.api.types.is_numeric_dtype(df[col])])

    if len(addi_list) > 1:
        selected_data.insert(0, addi, addi_list)

    # Convert to JSON string
    json_str = selected_data.to_json(orient='records', force_ascii=False)

    return json_str


def get_cur_date_str(format='%Y-%m-%d-%H-%M'):
    current_date = datetime.now()
    current_date_str = current_date.strftime(format)
    return current_date_str


def run_query(api_key, read_dir='./data', f_n='data_2024_03', save_dir='res', query_data_len=10, y_true_fp='./data/y_true.csv',
            conti_fn='cf', step_=1, need_diff=True):
    if conti_fn in f_n:
        print(f'{f_n} continue')
        return
    query_url = 'http://localhost:53080/v1/chat-messages'
    headers = {
        'Authorization': f"Bearer {api_key}"
    }
    cur_date_str = get_cur_date_str()
    fault_count = 0
    fault_in_query_data_len = False

    fault_dict = {'Q': [], 'A': []}
    normal_dict = {'Q': [], 'A': []}
    df = pd.read_csv(f'{read_dir}/{f_n}.csv')

    if not os.path.exists(f'./res/{save_dir}'):
        os.makedirs(f'./res/{save_dir}')

    if not os.path.exists(f'./res/{save_dir}/res_.csv'):
        with open(f'./res/{save_dir}/res_.csv', 'a+', encoding='utf-8-sig') as f:
            f.write('filename,y_true,y_pred,y_va\n')
    else:
        tmp_df = pd.read_csv(f'./res/{save_dir}/res_.csv')
        tmp_fn_list = list(tmp_df['filename'])
        if f_n in tmp_fn_list:
            print(f'{f_n} is in the result, skipped.')
            return

    y_va = 1
    y_df = pd.read_csv(y_true_fp)
    y_fn_list = list(y_df['filename'])
    y_true_list = list(y_df['y_true'])
    y_dict = {}
    for i in range(len(y_fn_list)):
        y_dict[y_fn_list[i]] = y_true_list[i]
    y_true_label = y_dict[f_n]
    if y_true_label == 1:
        step_ = query_data_len

    for i in tqdm.tqdm(range(0, len(df) - query_data_len, step_)):
        idx = i

        if fault_count > 0:
            break

        if idx % query_data_len == 0:
            fault_in_query_data_len = False

        str_ = df_to_json_idx_size(df, idx, query_data_len, need_diff=need_diff)
        payload = {
            'inputs': {},
            "query": str_,
            "response_mode": 'blocking',
            "user": f'test_{idx}'
        }
        rsp_data = None
        answer = None

        try:
            for i_ in range(3):
                response = requests.post(query_url, headers=headers, json=payload)
                rsp_data = response.json()
                status = rsp_data.get('status', None)
                answer = rsp_data.get('answer', None)
                if status != 400 or status is None:
                    break
                else:
                    print(f'{i_=}, {rsp_data=}, waiting 60s')
                    time.sleep(60)

            try:
                answer = json.loads(answer.replace('\n', '\\n'))
                msg = answer['msg']
                think = answer.get('think', '{}\n{}'.format(answer.get('think1'), answer.get('think2')))
                if y_va == 1 and (answer.get('think2') is not None or answer.get('think1') is not None or msg == '断煤故障'):
                    y_va = 0
            except Exception as e_:
                if 'Normal operation' in rsp_data['answer']:
                    msg = 'Normal operation'
                    think = rsp_data['answer']
                    if y_va == 1 and 'think2' in think:
                        y_va = 0
                elif '断煤故障' in rsp_data['answer']:
                    msg = '断煤故障'
                    think = rsp_data['answer']
                    y_va = 0
                else:
                    raise e_

            if msg == '断煤故障':
                if not fault_in_query_data_len:
                    fault_count += 1
                    fault_in_query_data_len = True

                fault_dict['Q'].append(str_)
                fault_dict['A'].append(think)

                with open(f'./res/{save_dir}/{f_n}_fa_{cur_date_str}.csv', 'a+', encoding='utf-8-sig') as f_:
                    f_.write('*' * 60)
                    f_.write('\n')
                    f_.write(f'{str_}\n')
                    f_.write(f'{think}\n')
                    f_.write(f'fault_count: {fault_count}\n')
                    f_.write('*' * 60)
                    f_.write('\n')
            else:
                normal_dict['Q'].append(str_)
                normal_dict['A'].append(think)
                with open(f'./res/{save_dir}/{f_n}_tr_{cur_date_str}.csv', 'a+', encoding='utf-8-sig') as f_:
                    f_.write('*' * 60)
                    f_.write('\n')
                    f_.write(f'{str_}\n')
                    f_.write(f'{think}\n')
                    f_.write('*' * 60)
                    f_.write('\n')
        except Exception as e_:
            tb_str = traceback.format_exc()
            print(f'{str_=}\n{rsp_data=}\n{answer=}\n{tb_str=}')

    with open(f'./res/{save_dir}/res_.csv', 'a+', encoding='utf-8-sig') as f:
        f.write('{},{},{},{}\n'.format(f_n, y_dict[f_n], int(fault_count < 1), y_va))

    df = pd.DataFrame(fault_dict)
    df.to_csv(f'./res/{save_dir}/{f_n}_fault_{cur_date_str}.csv', index=False, encoding='utf-8-sig')
    df = pd.DataFrame(normal_dict)
    df.to_csv(f'./res/{save_dir}/{f_n}_normal_{cur_date_str}.csv', index=False, encoding='utf-8-sig')


def run_va_10():
    read_dir = './data/test'
    query_data_len = 5
    api_key = ''
    for i in range(10):
        save_dir = f'valid_exp_{i}'
        for root, dirs, files in os.walk(read_dir):
            for file in files:
                if file.endswith('.csv'):
                    print(f'{file=}, {i=}')
                    run_query(api_key=api_key, read_dir=read_dir, f_n=file.replace('.csv', ''),
                              save_dir=save_dir, query_data_len=query_data_len, y_true_fp='./data/y_true.csv', step_=1)

        df = pd.read_csv(f'./res/{save_dir}/res_.csv', encoding='utf-8-sig')
        y_true = np.subtract(1, list(df['y_true']))
        y_pred = np.subtract(1, list(df['y_pred']))
        repo = classification_report(y_true, y_pred)
        acc_1 = accuracy_score(y_true, y_pred)
        p_1 = precision_score(y_true, y_pred)
        r_1 = recall_score(y_true, y_pred)
        f_1 = f1_score(y_true, y_pred)
        confusion_matrix_aft = confusion_matrix(y_true, y_pred)
        repo_2 = None
        if 'y_va' in df:
            y_va = np.subtract(1, list(df['y_va']))
            repo_2 = classification_report(y_true, y_va)
            acc_2 = accuracy_score(y_true, y_va)
            p_2 = precision_score(y_true, y_va)
            r_2 = recall_score(y_true, y_va)
            f_2 = f1_score(y_true, y_va)
            confusion_matrix_va = confusion_matrix(y_true, y_va)
        with open(f'./res/valid_exp.csv', 'a+', encoding='utf-8-sig') as f:
            f.write('*' * 60)
            f.write('\n')
            f.write(f'{i=}, aft\n{confusion_matrix_aft=}\n')
            f.write(repo)
            f.write(f'\n{acc_1=}, {p_1=}, {r_1=}, {f_1=}\n\n')
            if repo_2 is not None:
                f.write(f'{confusion_matrix_va=}\n')
                f.write(repo_2)
                f.write(f'\n{acc_2=}, {p_2=}, {r_2=}, {f_2=}\n')
            f.write('*' * 60)
            f.write('\n')


if __name__ == '__main__':
    run_va_10()
