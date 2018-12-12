# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


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


#ordem das cores na imagem
#(b, g, r) = imagem[0,0]

#ordem das coordenadas
#shape[0] é a altura, eixo y
#shape[1] é a largura, eixo x

if __name__ == '__main__':
    tipo_da_transformacao="fourier"
    # nome_da_imagem="mikasa_cosplay"
    nome_da_imagem = "rain_portrait"

    imagem = cv2.imread("imagens/"+nome_da_imagem+".jpg")
    print(imagem.shape)

    #cria uma imagem preta com o tamanho da imagem original para armazenar as alterações feitas na imagem original
    imagem_alterada = np.zeros((imagem.shape[0], imagem.shape[1], 3), dtype=np.uint8)

    cv2.imshow("original", imagem)

    #TESTES DE TRASFORMACOES

    #transforma a imagem em escalas de cinza
    imagem_alterada=cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    cv2.imshow("cinza", imagem_alterada)
    #print imagem_alterada
    ##########Rotações:
    centro=[imagem.shape[0]/2.0, imagem.shape[1]/2.0]

    #rotacao_direta(centro, matriz_rotacao(45), imagem, imagem_alterada)
    #rotacao_interpolada(centro, matriz_inversa_rotacao(3600), imagem, imagem_alterada)

    ###########Twirl:
    #alpha=np.pi/2
    #twirl(centro, imagem, imagem_alterada, alpha)

    #plot dos resultados da transformação

    ############Fourier:

    r_max=200
    r_min=100

    f = np.fft.fft2(imagem_alterada)
    fshift = np.fft.fftshift(f)

    #cv2.imshow("fshift", fshift.astype(np.uint8))

    #print "AAAAAAAAAAA"
    #print fshift.shape

    #magnitude_spectrum = (20*np.log(np.abs(fshift))).astype(np.uint8)


    #cv2.imshow(tipo_da_transformacao, magnitude_spectrum)

    passa_baixa(fshift, centro, r_max)
    # passa_alta(fshift, centro, r_min)
    # passa_banda(fshift, centro, r_min, r_max)


    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift).astype(np.uint8)


    #plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #plt.show()

    cv2.imshow("retorno", img_back)
    cv2.waitKey(0)

    #cv2.imwrite("transformacoes/"+tipo_da_transformacao+"_"+nome_da_imagem+".jpg", imagem_alterada)
