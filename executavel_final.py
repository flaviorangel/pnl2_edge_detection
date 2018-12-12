# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import edge_detection
import morphology
from matplotlib import pyplot as plt
import os


###########FUNÇÕES AUXILIARES#############

def create_degrade_simple(imagem):
    for y in range(0, imagem.shape[0]): #percorre linhas
        for x in range(0, imagem.shape[1]): #percorre colunas
            imagem[y, x] = (x%256,y%256,x%256)

def chamusca_imagem(imagem):
    for y in range(0, imagem.shape[0], 10): #percorre linhas
        for x in range(0, imagem.shape[1], 10): #percorre colunas
            imagem[y:y+5, x: x+5] = (0,255,255)

def slicing(imagem, cordx, cordy, cor):
    imagem[cordy[0]:cordy[1], cordx[0]:cordx[1]] = (cor[0], cor[1], cor[2])

def zoom_in(imagem, zoom):
    imagem = imagem[::zoom,::zoom]

def matriz_rotacao(angulo):
    radianos=np.radians(angulo)
    return [ [np.cos(radianos), np.sin(radianos)], [-np.sin(radianos), np.cos(radianos)] ]

def matriz_inversa_rotacao(angulo):
    radianos=np.radians(-angulo)
    return [ [np.cos(radianos), np.sin(radianos)], [-np.sin(radianos), np.cos(radianos)] ]

#retorna tupla de 3 dimenções das cores do pixel interpolado
def interpolacao_bilinear(posicao, imagem):
    x=posicao[1]
    y=posicao[0]
    x1=int(np.floor(posicao[1]))
    x2=int(np.ceil(posicao[1]))
    y1=int(np.floor(posicao[0]))
    y2=int(np.ceil(posicao[0]))
    q11=imagem[y1, x1]
    q21=imagem[y1, x2]
    q12=imagem[y2, x1]
    q22=imagem[y2, x2]

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


##################FUNÇÕES PRINCIPAIS#######################

def rotacao_direta(centro, matriz_rotacao, imagem, imagem_alterada):
    for y in range(0, imagem.shape[0]): #percorre linhas da imagem original
        for x in range(0, imagem.shape[1]): #percorre colunas da imagem original
            vetor_distancia=np.array([x-centro[1], y-centro[0]])
            vetor_deslocado=np.dot(matriz_rotacao, vetor_distancia)
            posicao_final=[int(vetor_deslocado[1]+centro[0]), int(vetor_deslocado[0]+centro[1])]
            if posicao_final[0]>=0 and posicao_final[0]<imagem_alterada.shape[0] and posicao_final[1]>=0 and posicao_final[1]<imagem_alterada.shape[1]:
                imagem_alterada[posicao_final[0], posicao_final[1]] = imagem[y,x]

def rotacao_interpolada(centro, matriz_inversa_rotacao, imagem, imagem_alterada):
    for y in range(0, imagem.shape[0]): #percorre linhas da imagem alterada
        for x in range(0, imagem.shape[1]): #percorre colunas da imagem alterada
            vetor_distancia=np.array([x-centro[1], y-centro[0]])

            #executa a multiplicacao a inversa da matriz de rotacao
            vetor_deslocado=np.dot(matriz_inversa_rotacao, vetor_distancia)

            #essa sera a posicao na imagem original
            posicao_final=[vetor_deslocado[1]+centro[0], vetor_deslocado[0]+centro[1]]
            if posicao_final[0]>=0 and posicao_final[0]<imagem_alterada.shape[0]-1 and posicao_final[1]>=0 and posicao_final[1]<imagem_alterada.shape[1]-1:
                #faz uma verificacao para ver se a posicao na imagem original crava em um pixel exato
                #caso isso aconteca nao sera necessario fazer uma interpolacao
                if posicao_final[0]%1==0 and posicao_final[1]%1==0:
                    imagem_alterada[y,x]=imagem[int(posicao_final[0]), int(posicao_final[1])]
                else:
                    cor=interpolacao_bilinear(posicao_final, imagem)
                    imagem_alterada[y,x]=cor

def twirl(centro, imagem, imagem_alterada, alpha):
    r_max=math.sqrt(centro[0]*centro[0]+centro[1]*centro[1])
    for y in range(0, imagem.shape[0]): #percorre linhas da imagem alterada
        for x in range(0, imagem.shape[1]): #percorre colunas da imagem alterada
            vetor_distancia=np.array([x-centro[1], y-centro[0]])
            r = math.sqrt(vetor_distancia[0]*vetor_distancia[0] + vetor_distancia[1]*vetor_distancia[1])
            if r>r_max:
                imagem_alterada[y,x]=imagem[y,x]
            else:
                beta=np.arctan2(vetor_distancia[0], vetor_distancia[1])+alpha*((r_max-r)/r_max)
                posicao_final=[centro[0]+r*np.sin(beta), centro[1]+r*np.cos(beta)]
                if posicao_final[0]>=0 and posicao_final[0]<imagem_alterada.shape[0]-1 and posicao_final[1]>=0 and posicao_final[1]<imagem_alterada.shape[1]-1:
                    #faz uma verificacao para ver se a posicao na imagem original crava em um pixel exato
                    #caso isso aconteca nao sera necessario fazer uma interpolacao
                    if posicao_final[0]%1==0 and posicao_final[1]%1==0:
                        imagem_alterada[y,x]=imagem[int(posicao_final[0]), int(posicao_final[1])]
                    else:
                        cor=interpolacao_bilinear(posicao_final, imagem)
                        imagem_alterada[y,x]=cor

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


def passa_baixa(imagem_frequencia, centro, r_max):
    for y in range(0, imagem_frequencia.shape[0]): #percorre linhas da imagem alterada
        for x in range(0, imagem_frequencia.shape[1]): #percorre colunas da imagem alterada
            vetor_distancia=np.array([x-centro[1], y-centro[0]])
            r = math.sqrt(vetor_distancia[0]*vetor_distancia[0] + vetor_distancia[1]*vetor_distancia[1])
            if r>r_max:
                imagem_frequencia[y,x]=0

def passa_alta(imagem_frequencia, centro, r_max):
    for y in range(0, imagem_frequencia.shape[0]): #percorre linhas da imagem alterada
        for x in range(0, imagem_frequencia.shape[1]): #percorre colunas da imagem alterada
            vetor_distancia=np.array([x-centro[1], y-centro[0]])
            r = math.sqrt(vetor_distancia[0]*vetor_distancia[0] + vetor_distancia[1]*vetor_distancia[1])
            if r<r_max:
                imagem_frequencia[y,x]=0

def passa_banda(imagem_frequencia, centro, r_min, r_max):
    for y in range(0, imagem_frequencia.shape[0]): #percorre linhas da imagem alterada
        for x in range(0, imagem_frequencia.shape[1]): #percorre colunas da imagem alterada
            vetor_distancia=np.array([x-centro[1], y-centro[0]])
            r = math.sqrt(vetor_distancia[0]*vetor_distancia[0] + vetor_distancia[1]*vetor_distancia[1])
            if r>r_max:
                imagem_frequencia[y,x]=0
            elif r<r_min:
                imagem_frequencia[y,x]=0

def verifica_e_remove_pixel_proximo(x,y,imagem):
    for x1 in range(x-1,x-2):
        for y1 in range(y-1, y-2):
            imagem[y1,x1]=0

def Fourier(imagem_alterada, passa, r_max=100, r_min=50 ):
    f = np.fft.fft2(imagem_alterada)
    fshift = np.fft.fftshift(f)

    #cv2.imshow("fshift", fshift.astype(np.uint8))

    #print "AAAAAAAAAAA"
    #print fshift.shape

    #magnitude_spectrum = (20*np.log(np.abs(fshift))).astype(np.uint8)


    #cv2.imshow(tipo_da_transformacao, magnitude_spectrum)
    if passa=="passa_baixa":
        passa_baixa(fshift, centro, r_max)
    elif passa=="passa_alta":
        passa_alta(fshift, centro, r_min)
    else:
        passa_banda(fshift, centro, r_min, r_max)


    f_ishift = np.fft.ifftshift(fshift)
    imagem_alterada = np.fft.ifft2(f_ishift).astype(np.uint8)
    return imagem_alterada


def binarizacao(imagem, treshhold=50):
    #substitui os valores da imagem por 1 e 0.
    for y in range(0, imagem.shape[0]): #percorre linhas da imagem alterada
        for x in range(0, imagem.shape[1]): #percorre colunas da imagem alterada
            if imagem[y,x] < treshhold:
                imagem[y,x]=0
            else:
                imagem[y,x]=1


def retorno_binarizacao(imagem):
    #substitui os valores da imagem por 255 e 0.
    for y in range(0, imagem.shape[0]): #percorre linhas da imagem alterada
        for x in range(0, imagem.shape[1]): #percorre colunas da imagem alterada
            if imagem[y,x]:
                imagem[y,x]=255


def Hough(imagem_alterada, tipo):
    #imagem que é recebida por essa funçao já é uma Extracao de contorno
    if tipo=="normal":
        lines = cv2.HoughLines(imagem_alterada,1,np.pi/180,260)

        #print lines

        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(imagem_alterada,(x1,y1),(x2,y2),(255),2)
    else:
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(imagem_alterada,1,np.pi/180,100,minLineLength,maxLineGap)
        #print lines
        for line in lines:
            for x1,y1,x2,y2 in line:
                for x in range(x1,x2):
                    y=((y2-y1)+0.0/(x2-x1)+0.0)*(x-x1)+y1
                    verifica_e_remove_pixel_proximo(x,int(y),imagem_alterada)
                cv2.line(imagem_alterada ,(x1,y1),(x2,y2),(255),2)


#ordem das cores na imagem
#(b, g, r) = imagem[0,0]

#ordem das coordenadas
#shape[0] é a altura, eixo y
#shape[1] é a largura, eixo x

if __name__ == '__main__':

    #setups de variaveis que irão para teste
    nome_das_imagens=[f for f in os.listdir("imagens")]
    passas=["passa_baixa","passa_alta","passa_banda"]
    tipo_da_transformacao="Combinadas"
    treshholds=[50, 100, 150]
    #nome_da_imagem="janela2"


    for nome_da_imagem in nome_das_imagens:
        for passa in passas:
            for treshhold in treshholds:
                pasta=nome_da_imagem.split(".")[0]
                imagem = cv2.imread("imagens/"+nome_da_imagem)

                # cv2.imshow("original_com_cor", imagem)
                # cv2.waitKey(0)

                endereco = os.getcwd() + "/transformacoes/" + pasta + "/"

                if not os.path.exists(endereco):
                    os.makedirs(endereco)

                endereco += str(treshhold) + "_" + passa + "_" + nome_da_imagem

                #transforma a imagem em escalas de cinza
                imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

                cv2.imshow("original", imagem)
                # cv2.waitKey(0)

                #print imagem_alterada

                centro=[imagem.shape[0]/2.0, imagem.shape[1]/2.0]

                ########INICIO DAS TRASFORMACOES############

                ############Extração de contorno

                # cria uma imagem preta com o tamanho da imagem original para armazenar as alterações feitas na imagem original
                imagem_alterada = np.zeros((imagem.shape[0], imagem.shape[1]), dtype=np.uint8)

                sobel_treshhold(imagem, imagem_alterada, treshhold)

                cv2.imshow("edge", imagem_alterada)
                # cv2.waitKey(0)

                edge_detection.save_image(imagem_alterada, "_edge", endereco)
                # cv2.imwrite(os.getcwd() + "/transformacoes/" + pasta + "/" + str(
                #     treshhold) + "_" + passa + "_" + nome_da_imagem + "_edge_detection",
                #             imagem_alterada)

                ############Fourier:
                imagem = Fourier(imagem_alterada, passa)

                cv2.imshow("fourier", imagem)
                # cv2.waitKey(0)

                edge_detection.save_image(imagem, "_fourier", endereco)
                # cv2.imwrite(os.getcwd() + "/transformacoes/" + pasta + "/" + str(
                #     treshhold) + "_" + passa + "_" + nome_da_imagem + "_fourier",
                #             imagem)


                ##############Esqueletizacao

                # elemento estruturante
                se_3 = np.ones(shape=(3, 3))
                se_3[0, 0] = 0
                se_3[2, 0] = 0
                se_3[0, 2] = 0
                se_3[2, 2] = 0
                imagem = morphology.skeletonization(imagem, se_3)

                cv2.imshow("skeleton", imagem)
                # cv2.waitKey(0)

                edge_detection.save_image(imagem, "_edge", endereco)
                # cv2.imwrite(os.getcwd() + "/transformacoes/" + pasta + "/" + str(treshhold) + "_" + passa + "_" + nome_da_imagem + "_skeletonization",
                #             imagem)


                ##############Hough:
                binarizacao(imagem)

                Hough(imagem, "P")

                retorno_binarizacao(imagem)

                ###INSERIR

                cv2.imshow("retorno", imagem)
                cv2.waitKey(0)

                cv2.imwrite(os.getcwd() + "/transformacoes/"+pasta+"/"+str(treshhold)+"_"+passa+"_"+nome_da_imagem, imagem)
