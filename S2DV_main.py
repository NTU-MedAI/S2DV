'''
以下是服务器python环境上需要安装的python包
尤其注意rdkit
'''
import numpy as np
import pickle as pkl
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


#这个是streamlit Web UI框架 仅仅展示期望效果，服务器可不添加这个packages
import streamlit as st



def get_ECFP(mol, radio):
    ECFPs = mol2alt_sentence(mol, radio)
    if len(ECFPs) % (radio + 1) != 0:
        ECFPs = ECFPs[:-(len(ECFPs) % (radio + 1))]
    ECFP_by_radio = list((np.array(ECFPs).reshape((int(len(ECFPs) / (radio + 1)), (radio + 1))))[:, radio])
    return ECFP_by_radio


def mol2alt_sentence(mol, radius):
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)

def get_sentence_vec(tokens,embedding,token_dict):
    #一句的tokens 利用 embedding 转为 句子的vec 用均值计算句子vec
    feature_vec = np.zeros(512) #创建1维全0数组
    n=0
    for token in tokens:
        if token in token_dict:
            feature_vec = np.add(feature_vec, embedding[token_dict[token]])
        else:n+=1
    sent_vec = np.divide(feature_vec, len(tokens)-n)
    #print(' {} token not found in tokens'.format(n))
    return sent_vec

def vec_predict(vec,models,ML_model):
    #这边功能可能需要一些环境配置，这些我可以自己来，这边随意先模拟一个输出就行

    #读取一个存放的数据表 pkl格式,不大
    #dict = pkl.load(open('./abcd.pkl','rb+'))
    for model_name,model in models:
        if model_name == ML_model:
            print(vec)
            print(len(vec))
            Y_predict = model.predict(vec.reshape(1, -1))
            Y_predict_p = model.predict_proba(vec.reshape(1, -1))[:, 1]

    # predict_result = {
    #     'predict_HBV_label':True,
    #     'predict_HepG2_label':False,
    #     'If_good_drug':True
    # }
    return Y_predict, Y_predict_p

def main(smiles = 'Nc1cc(OCCOCP(=O)(O)O)nc(N)n1'):

    #smiles = 'Nc1cc(OCCOCP(=O)(O)O)nc(N)n1'    #这个参数需要能从网页的文本框获取
    mol = Chem.MolFromSmiles(smiles)

    #绘制图片 看情况，如果不麻烦就显示一下bmp或者png图片 img.save
    img = Draw.MolsToGridImage([mol], molsPerRow=3, subImgSize=(300, 400))
    # img.save('/media/ntu/0698079098077E05/sylershao/药物分子活性预测/data_process/smiles2graph.bmp')
    #img.save('/media/ntu/0698079098077E05/sylershao/药物分子活性预测/data_process/smiles2graph.png')

    ECFP = get_ECFP(mol, 1)

    model_root = './Model_for_web/'
    HepG2_model = pkl.load(
        open(os.path.join(model_root, 'HepG2.ECFP.models.pkl'), 'rb+'))
    HepG2_token = pkl.load(
        open(os.path.join(model_root, 'HepG2_token.pkl'), 'rb+'))
    HepG2_emb = pkl.load(
        open(os.path.join(model_root, 'HepG2_emb.pkl'), 'rb+'))

    HBV_model = pkl.load(
        open(os.path.join(model_root, 'HBV.ECFP.models.pkl'), 'rb+'))
    HBV_token = pkl.load(
        open(os.path.join(model_root, 'HBV_token.pkl'), 'rb+'))
    HBV_emb = pkl.load(
        open(os.path.join(model_root, 'HBV_emb.pkl'), 'rb+'))

    HBV_vec = get_sentence_vec(ECFP,HBV_emb,HBV_token)
    HBV_predict, _ = vec_predict(HBV_vec,HBV_model,'XGBoost')

    HepG2_vec = get_sentence_vec(ECFP, HepG2_emb, HepG2_token)
    HepG2_predict, _ = vec_predict(HepG2_vec, HepG2_model, 'SVM')


    #输出判断
    conclusion = ''
    HBV_result = ''
    HepG2_result = ''
    if HBV_predict:
        HBV_result = '抑制率高，IC50低于1uM'
    else:
        HBV_result = '抑制率低，IC50高于1uM'

    if HepG2_predict:
        HepG2_result = '毒性高，CC50低于 30uM'
    else:
        HepG2_result = '毒性低，CC50高于 30uM'

    if HBV_predict and not HepG2_predict: conclusion = '具有作为药物潜力'
    if HBV_predict and HepG2_predict: conclusion = '具有作为药物潜力，但是需要微调以降低药物毒性'
    if not HBV_predict and HepG2_predict:conclusion = '不具备制作成为药物的潜力'
    if not HBV_predict and not HepG2_predict: conclusion = '不具备制作成为药物的潜力'

    result4webshow = '输入的SMILES: {}  \r\n' \
                     'HBV的抑制效果预测为：{}  \r\n' \
                     'HepG2的毒性预测为: {}   \r\n' \
                     '是否能够作为潜在HBV药物: {}'.\
        format(
        smiles,
        HBV_result,
        HepG2_result,
        conclusion)
    print(result4webshow) #只需要输出这句话就可以
    return result4webshow,img

def web_demo(streamlit=None):
    #用streamlit做了个临时效果
    st.title('S2DV:predict anti-HBV drugs')
    smiles = st.text_input('input SMILES', value='e.g. Nc1cc(OCCOCP(=O)(O)O)nc(N)n1', max_chars=None, key=None, type='default', help=None, autocomplete=None,
                         on_change=None, args=None, kwargs=None)
    if st.button('Predict Drugs'):
        text,img = main(smiles)
        st.image(img,use_column_width=True,caption='Mol_structure')
        st.text(text)

if __name__ == '__main__':
    web_demo() # 安装streamlit后 shell 运行 streamlit run Website_function.py