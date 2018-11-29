# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math

#########FUNÇÕES DE SUAVIZAÇÃO##############
def filtro_mediano(imagem):
    imagem_alterada = np.zeros((imagem.shape[0], imagem.shape[1], 3), dtype=np.uint8)
    for y in range(1,imagem_alterada.shape[0]-1):
        for x in range(1,imagem_alterada.shape[1]-1):
            imagem_alterada[y,x]=np.median(imagem[y-1:y+2, x-1:x+2])
    return imagem_alterada


def filtro_gaussiano(imagem):
    imagem_alterada = np.zeros((imagem.shape[0], imagem.shape[1], 3), dtype=np.uint8)
    mascara=[[1,2,1],[2,4,2],[1,2,1]]
    for y in range(1,imagem_alterada.shape[0]-1):
        for x in range(1,imagem_alterada.shape[1]-1):
            cor=np.sum(mascara*imagem[y-1:y+2, x-1:x+2])/16
            imagem_alterada[y,x]=cor
    return imagem_alterada





###############FUNÇÕES DE CONTORNO#####################
def previt(imagem, imagem_alterada):
    #as mascaras parescem estar trocadas, mas nossa imagem a primeira coordenada é o y, e a segunda é o x, pelo cv2
    #logo para ajustar, eu tive que inverter as matrizes para se adequar aos dados
    mascara=[0]*8
    mascara[0]= [[-1,-1,-1],
                [0,0,0],
                [1,1,1]]

    mascara[1]= [[0,-1,-1],
                [1,0,-1],
                [1,1,0]]

    mascara[2]= [[1,0,-1],
                [1,0,-1],
                [1,0,-1]]

    mascara[3]= [[1,1,0],
                [1,0,-1],
                [0,-1,-1]]

    mascara[4]= [[1,1,1],
                [0,0,0],
                [-1,-1,-1]]

    mascara[5]= [[0,1,1],
                [-1,0,1],
                [-1,-1,0]]

    mascara[6]= [[-1,0,1],
                [-1,0,1],
                [-1,0,1]]

    mascara[7]= [[-1,-1,0],
                [-1,0,1],
                [0,1,1]]
    for y in range(1,imagem_alterada.shape[0]-1):
        for x in range(1,imagem_alterada.shape[1]-1):
            vetor_de_gradiente=[]
            for m in mascara:
                vetor_de_gradiente.append(np.sum(m*imagem[y-1:y+2, x-1:x+2]))
            #pega o maior valor de gradiente encontrado pelas rotaçoes das mascaras
            imagem_alterada[y,x]=sorted(vetor_de_gradiente)[-1]

def previt_treshhold(imagem, imagem_alterada, treshhold):
    #as mascaras parescem estar trocadas, mas nossa imagem a primeira coordenada é o y, e a segunda é o x, pelo cv2
    #logo para ajustar, eu tive que inverter as matrizes para se adequar aos dados
    mascara=[0]*8
    mascara[0]= [[-1,-1,-1],
                [0,0,0],
                [1,1,1]]

    mascara[1]= [[0,-1,-1],
                [1,0,-1],
                [1,1,0]]

    mascara[2]= [[1,0,-1],
                [1,0,-1],
                [1,0,-1]]

    mascara[3]= [[1,1,0],
                [1,0,-1],
                [0,-1,-1]]

    mascara[4]= [[1,1,1],
                [0,0,0],
                [-1,-1,-1]]

    mascara[5]= [[0,1,1],
                [-1,0,1],
                [-1,-1,0]]

    mascara[6]= [[-1,0,1],
                [-1,0,1],
                [-1,0,1]]

    mascara[7]= [[-1,-1,0],
                [-1,0,1],
                [0,1,1]]
    for y in range(1,imagem_alterada.shape[0]-1):
        for x in range(1,imagem_alterada.shape[1]-1):
            vetor_de_gradiente=[]
            for m in mascara:
                vetor_de_gradiente.append(np.sum(m*imagem[y-1:y+2, x-1:x+2]))
            #pega o maior valor de gradiente encontrado pelas rotaçoes das mascaras
            max_grad=sorted(vetor_de_gradiente)[-1]
            if max_grad>=treshhold:
                imagem_alterada[y,x]=255
            else:
                imagem_alterada[y,x]=0

def sobel(imagem, imagem_alterada):
    #as mascaras parescem estar trocadas, mas nossa imagem a primeira coordenada é o y, e a segunda é o x, pelo cv2
    #logo para ajustar, eu tive que inverter as matrizes para se adequar aos dados
    mascaraX=[[-1, -2, -1],[0,0,0],[1,2,1]]
    mascaraY=[[1,0,-1],[2,0,-2],[1,0,-1]]
    for y in range(1,imagem_alterada.shape[0]-1):
        for x in range(1,imagem_alterada.shape[1]-1):
            gx=np.sum(mascaraX*imagem[y-1:y+2, x-1:x+2])
            gy=np.sum(mascaraY*imagem[y-1:y+2, x-1:x+2])
            grad=np.sqrt(gx*gx+gy*gy)
            imagem_alterada[y,x]=grad

def sobel_treshhold(imagem, imagem_alterada, treshhold):
    #as mascaras parescem estar trocadas, mas nossa imagem a primeira coordenada é o y, e a segunda é o x, pelo cv2
    #logo para ajustar, eu tive que inverter as matrizes para se adequar aos dados
    mascaraX=[[-1, -2, -1],[0,0,0],[1,2,1]]
    mascaraY=[[1,0,-1],[2,0,-2],[1,0,-1]]
    for y in range(1,imagem_alterada.shape[0]-1):
        for x in range(1,imagem_alterada.shape[1]-1):
            gx=np.sum(mascaraX*imagem[y-1:y+2, x-1:x+2])
            gy=np.sum(mascaraY*imagem[y-1:y+2, x-1:x+2])
            grad=np.sqrt(gx*gx+gy*gy)
            if grad>=treshhold:
                imagem_alterada[y,x]=255
            else:
                imagem_alterada[y,x]=0


def canny():
    return "TODO"


#Para aplicar uma mascara m1 em uma matriz de bits m2 basta fazer m1*m2 (o numpy array faz multiplicacao coordenada por coordenada)
#Para pegar a soma de todos os elementos de um array basta np.sum(array)
if __name__ == '__main__':
    tipo_do_filtro="previt_treshhold_150_com_filtro_gaussiano"
    nome_da_imagem="titan_forest"

    imagem = cv2.imread("imagens/"+nome_da_imagem+".jpg")

    #deixa a imagem em tons de cinza para fascilitar
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_alterada = np.zeros((imagem.shape[0], imagem.shape[1], 3), dtype=np.uint8)

    cv2.imshow("original", imagem)

    ############TESTE DE FUNÇÕES###########


    #imagem_alterada=filtro_mediano(imagem)
    #imagem_alterada=filtro_gaussiano(imagem)
    #previt(imagem, imagem_alterada)
    previt_treshhold(filtro_gaussiano(imagem), imagem_alterada, 150)
    #sobel(imagem, imagem_alterada)
    #sobel_treshhold(filtro_gaussiano(imagem), imagem_alterada, 150)

    cv2.imshow("transformada", imagem_alterada)

    cv2.waitKey(0)

    cv2.imwrite("filtros/"+tipo_do_filtro+"_"+nome_da_imagem+".jpg", imagem_alterada)
