import sys
import os
import pyglet
import pyglet.media as media
import numpy as np
import random

import libs.transformations as tr
import libs.shaders as sh
import libs.scene_graph as sg
import libs.basic_shapes as bs
import libs.lighting_shaders as ls

from libs.gpu_shape import createGPUShape
from libs.obj_handler import read_OBJ
from libs.obj_handler import read_OBJ2
from libs.assets_path import getAssetPath

from OpenGL.GL import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assets de Objetos 
ASSETS = {
    "nave": getAssetPath("nave.obj"),
    "enemy": getAssetPath("enemy.obj"),
    "meteorito": getAssetPath("meteorite.obj"),
    "sonda": getAssetPath("sonda.obj"),
    "fondo": getAssetPath("fondo.obj"),
    "textura": getAssetPath("fondo.jpg")}

class Nave:
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z

        self.theta = np.pi/2
        self.phi = -np.pi/2

        self.X_speed = 0.0
        self.Y_speed = 0.0
        self.Z_speed = 0.0

    def update(self, dt):
        self.theta += self.Y_speed * dt
        self.phi += self.Z_speed * dt

        self.X += dt * self.X_speed * np.sin(self.theta) * np.sin(self.phi)
        self.Y += dt * self.X_speed * np.cos(self.theta)
        self.Y = max(self.Y,0)
        self.Z += dt * self.X_speed * np.sin(self.theta) * np.cos(self.phi)

        self.transform = tr.matmul([
            tr.translate(self.X, self.Y, self.Z),
            tr.scale(0.2, 0.2, 0.2),
            tr.rotationY(self.phi),
            tr.rotationX(self.theta - np.pi/2),
        ])

# Se define la Sombra
class Sombra:
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z

        self.theta = np.pi/2
        self.phi = -np.pi/2

        self.X_speed = 0.0
        self.Y_speed = 0.0
        self.Z_speed = 0.0

    def update(self, dt):
        self.theta += self.Y_speed * dt
        self.phi += self.Z_speed * dt

        self.X += dt * self.X_speed * np.sin(self.theta) * np.sin(self.phi)
        self.Z += dt * self.X_speed * np.sin(self.theta) * np.cos(self.phi)

        self.transform = tr.matmul([
            tr.translate(self.X, self.Y, self.Z),
            tr.scale(0.2, 0.01, 0.2),
            tr.rotationY(self.phi),
            tr.rotationX(self.theta - np.pi/2),
        ])
        
# Se define al Enemigo
class Enemigo:
    def __init__(self,x,y,z):
        self.X = x
        self.Y = y
        self.Z = z
        self.x = x
        self.z = z
        self.alpha = 0
        self.term = random.randint(1,5)
        self.term2 = random.randint(1,5)

    def update(self,dt):
        self.alpha += np.pi/8 * dt
        self.X =  self.term*np.cos(self.term*self.alpha)
        self.Z =   self.term2*np.cos(self.term2*self.alpha)
        self.transform = tr.matmul([tr.rotationY(-np.pi/2),
                                    tr.translate(self.X + self.x, self.Y, self.Z + self.z), 
                                    tr.scale(0.5, 0.5, 0.5)])


# Se define el Meteorito
class Meteoro:
    def __init__(self,x,y,z):
        self.X = x
        self.Y = y
        self.Z = z
        self.phi = np.pi
        self.term = random.randint(-5,5)
        
    def update(self, dt):
        self.phi += self.term * np.pi/10 * dt
        self.transform = tr.matmul([tr.rotationY(self.phi),
                                    tr.translate(self.X, self.Y, self.Z),
                                    tr.scale(0.01,0.01,0.01)])

class Sonda:
    def __init__(self,x,y,z):
        self.X = x
        self.Y = y
        self.Z = z
        self.x = x
        self.z = z
        self.alpha = 0
        self.term = random.randint(1,5)
        self.term2 = random.randint(1,5)

    def update(self,dt):
        self.alpha += np.pi/8 * dt
        self.X =  self.term*np.cos(self.term*self.alpha)
        self.Z =   self.term2*np.cos(self.term2*self.alpha)
        self.transform = tr.matmul([tr.rotationY(np.pi/2),
                                    tr.translate(self.X + self.x, self.Y, self.Z + self.z), 
                                    tr.scale(0.1, 0.1, 0.1)])


# Se define el MapaAA
class Mapa:
    def __init__(self,x,y,z):
        self.X = x
        self.Y = y
        self.Z = z
        self.transform = tr.matmul([tr.translate(self.X + 5, self.Y - 2, self.Z + 20), 
                                    tr.scale(-100, 0.05, 10)])

# Se define el Controller
class Controller(pyglet.window.Window):
    def __init__(self, width, height, title="Naves y Estrellas 3D"):
        super().__init__(width, height, title)
        self.total_time = 0.0
        self.pipeline = ls.SimpleGouraudShaderProgram()

# Se ejecuta el controller
controller = Controller(800,800)

# Se ejecutan los objetos
Naves = [Nave(1,2,0), Nave(0,2,-1), Nave(0,2,1), Sombra(1,-0.9,0), Sombra(0,-0.9,-1), Sombra(0,-0.9,1)]
Enemigos = [Enemigo(-5,2,-15), Enemigo(5,2,-15)]
Meteoros = [Meteoro(random.randint(-15,15),random.randint(0,5),random.randint(-15,15)) for i in range(10)]
Sondas = [Sonda(random.randint(-10,10),random.randint(0,5),random.randint(8,10)) for i in range(6)]
cubo = Mapa(0,-1,0)

# Se define la cámara
class Camera:
    def __init__(self):
        self.at = np.array([0.0, 0.0, 0.0])
        self.eye = np.array([-8.0, 8.0, 9.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.projection = tr.ortho(-8, 8, -8, 8, 0.1, 100)

    def update(self, dt):
        pos = np.array([Naves[0].X, Naves[0].Y, Naves[0].Z])
        self.at = pos
        self.eye = np.array([-8.0, 8.0, 9.0]) + pos

# Se ejecuta la cámara
camera = Camera()

# Se crean en la GPU
gpuNave = createGPUShape(controller.pipeline, read_OBJ(ASSETS["nave"],(1,0.1,0.1)))
gpuNaveLider = createGPUShape(controller.pipeline, read_OBJ(ASSETS["nave"],(0.9,0.9,0.9)))
gpuSombraNave = createGPUShape(controller.pipeline, read_OBJ(ASSETS["nave"],(0,0,0)))
gpuEnemigo = createGPUShape(controller.pipeline, read_OBJ(ASSETS["enemy"],(0.55,0.28,0.14)))
gpuMeteoro = createGPUShape(controller.pipeline, read_OBJ(ASSETS["meteorito"],(0.2,0.2,0.2)))
gpuSonda = createGPUShape(controller.pipeline, read_OBJ(ASSETS["sonda"],(1,1,0.1)))
gpuCubo = createGPUShape(controller.pipeline, read_OBJ(ASSETS["fondo"],(0.75,0.75,0.75)))

# Grafo de Escena
## Se crean los nodos
nave1 = sg.SceneGraphNode("nave1")
nave2 = sg.SceneGraphNode("nave2")
nave3 = sg.SceneGraphNode("nave3")
sombra1 = sg.SceneGraphNode("sombra1")
sombra2 = sg.SceneGraphNode("sombra2")
sombra3 = sg.SceneGraphNode("sombra3")
enemigo1 = sg.SceneGraphNode("enemigo1")
enemigo2 = sg.SceneGraphNode("enemigo2")
meteoro1 = sg.SceneGraphNode("meteoro1")
meteoro2 = sg.SceneGraphNode("meteoro2")
meteoro3 = sg.SceneGraphNode("meteoro3")
meteoro4 = sg.SceneGraphNode("meteoro4")
meteoro5 = sg.SceneGraphNode("meteoro5")
meteoro6 = sg.SceneGraphNode("meteoro6")
meteoro7 = sg.SceneGraphNode("meteoro7")
meteoro8 = sg.SceneGraphNode("meteoro8")
meteoro9 = sg.SceneGraphNode("meteoro9")
meteoro10 = sg.SceneGraphNode("meteoro10")
sonda1 = sg.SceneGraphNode("sonda1")
sonda2 = sg.SceneGraphNode("sonda2")
sonda3 = sg.SceneGraphNode("sonda3")
sonda4 = sg.SceneGraphNode("sonda4")
sonda5 = sg.SceneGraphNode("sonda5")
sonda6 = sg.SceneGraphNode("sonda6")
map = sg.SceneGraphNode("cubo")
## Se añaden los hijos a los nodos
nave1.childs += [gpuNaveLider]
nave2.childs += [gpuNave]
nave3.childs += [gpuNave]
sombra1.childs += [gpuSombraNave]
sombra2.childs += [gpuSombraNave]
sombra3.childs += [gpuSombraNave]
enemigo1.childs += [gpuEnemigo]
enemigo2.childs += [gpuEnemigo]
meteoro1.childs += [gpuMeteoro]
meteoro2.childs += [gpuMeteoro]
meteoro3.childs += [gpuMeteoro]
meteoro4.childs += [gpuMeteoro]
meteoro5.childs += [gpuMeteoro]
meteoro6.childs += [gpuMeteoro]
meteoro7.childs += [gpuMeteoro]
meteoro8.childs += [gpuMeteoro]
meteoro9.childs += [gpuMeteoro]
meteoro10.childs += [gpuMeteoro]
sonda1.childs += [gpuSonda]
sonda2.childs += [gpuSonda]
sonda3.childs += [gpuSonda]
sonda4.childs += [gpuSonda]
sonda5.childs += [gpuSonda]
sonda6.childs += [gpuSonda]
map.childs += [gpuCubo]
## Se crea el nodo principal
squad = sg.SceneGraphNode("squad")
squad.childs += [nave1]
squad.childs += [nave2]
squad.childs += [nave3]
squad.childs += [sombra1]
squad.childs += [sombra2]
squad.childs += [sombra3]
squad.childs += [enemigo1]
squad.childs += [enemigo2]
squad.childs += [meteoro1]
squad.childs += [meteoro2]
squad.childs += [meteoro3]
squad.childs += [meteoro4]
squad.childs += [meteoro5]
squad.childs += [meteoro6]
squad.childs += [meteoro7]
squad.childs += [meteoro8]
squad.childs += [meteoro9]
squad.childs += [meteoro10]
squad.childs += [sonda1]
squad.childs += [sonda2]
squad.childs += [sonda3]
squad.childs += [sonda4]
squad.childs += [sonda5]
squad.childs += [sonda6]
squad.childs += [map]

# Creación de estrellas
num_estrellas = 100
for i in range(num_estrellas):
    # Crear cubo de color aleatorio
    gpuCuboEstrella = createGPUShape(controller.pipeline, bs.createColorNormalsCube(1, 1, 1))

    # Posicionar cubo en una posición aleatoria
    x = random.uniform(-50, 50)
    y = random.uniform(-50, -1)
    z = random.uniform(-50, 50)

    # Crear nodo de la estrella y agregarlo al grafo de escena
    estrella = sg.SceneGraphNode("estrella" + str(i))
    estrella.transform = tr.matmul([tr.translate(x, y, z), tr.scale(0.05, 0.05, 0.05)])
    estrella.childs += [gpuCuboEstrella]
    squad.childs += [estrella]




# Se define el fondo
glClearColor(0,0,0, 1.0)
glEnable(GL_DEPTH_TEST)
glUseProgram(controller.pipeline.shaderProgram)



# Se hace la iluminación
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Ka"), 0.5, 0.5, 0.5)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "lightPosition"), 0 ,8, 0)
glUniform1ui(glGetUniformLocation(controller.pipeline.shaderProgram, "shininess"), 100)
glUniform1f(glGetUniformLocation(controller.pipeline.shaderProgram, "constantAttenuation"), 0.0001)
glUniform1f(glGetUniformLocation(controller.pipeline.shaderProgram, "linearAttenuation"), 0.03)
glUniform1f(glGetUniformLocation(controller.pipeline.shaderProgram, "quadraticAttenuation"), 0.01)

# What happens when the user presses these keys
@controller.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.W:
        for nave in Naves:
            nave.X_speed = -5
    elif symbol == pyglet.window.key.S:
        for nave in Naves:
            nave.X_speed = 5
    elif symbol == pyglet.window.key.A:
        for nave in Naves:
            nave.Z_speed = 5
    elif symbol == pyglet.window.key.D:
        for nave in Naves:
            nave.Z_speed = -5

# What happens when the user releases these keys
@controller.event
def on_key_release(symbol, modifiers):
    if symbol == pyglet.window.key.W:
        for nave in Naves:
            nave.X_speed = 0
    elif symbol == pyglet.window.key.S:
        for nave in Naves:
            nave.X_speed = 0
    elif symbol == pyglet.window.key.A:
        for nave in Naves:
            nave.Z_speed = 0
    elif symbol == pyglet.window.key.D:
        for nave in Naves:
            nave.Z_speed = 0

@controller.event
def on_mouse_motion(x, y, dx, dy):
    for nave in Naves:
        nave.theta = ((y-dy)/800)*np.pi

music = pyglet.resource.media('musica.mp3')
music.play()


# Se dibuja
@controller.event
def on_draw():
    controller.clear()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    view = tr.lookAt(
        camera.eye,
        camera.at,
        camera.up
    )

    glUniformMatrix4fv(glGetUniformLocation(controller.pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
    glUniformMatrix4fv(glGetUniformLocation(controller.pipeline.shaderProgram, "projection"), 1, GL_TRUE, camera.projection)
    glUniformMatrix4fv(glGetUniformLocation(controller.pipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    sg.drawSceneGraphNode(squad, controller.pipeline, "model")

def update(dt, controller):
    controller.total_time += dt


    for nave in Naves:
        nave.update(dt)
    for enemigo in Enemigos:
        enemigo.update(dt)
    for meteor in Meteoros:
        meteor.update(dt)
    for sonda in Sondas:
        sonda.update(dt)
    
    camera.update(dt)

    nave1.transform = Naves[0].transform
    nave2.transform = Naves[1].transform
    nave3.transform = Naves[2].transform
    sombra1.transform = Naves[3].transform
    sombra2.transform = Naves[4].transform
    sombra3.transform = Naves[5].transform
    enemigo1.transform = Enemigos[0].transform
    enemigo2.transform = Enemigos[1].transform
    meteoro1.transform = Meteoros[0].transform
    meteoro2.transform = Meteoros[1].transform
    meteoro3.transform = Meteoros[2].transform
    meteoro4.transform = Meteoros[3].transform
    meteoro5.transform = Meteoros[4].transform
    meteoro6.transform = Meteoros[5].transform
    meteoro7.transform = Meteoros[6].transform
    meteoro8.transform = Meteoros[7].transform
    meteoro9.transform = Meteoros[8].transform
    meteoro10.transform = Meteoros[9].transform
    sonda1.transform = Sondas[0].transform
    sonda2.transform = Sondas[1].transform
    sonda3.transform = Sondas[2].transform
    sonda4.transform = Sondas[3].transform
    sonda5.transform = Sondas[4].transform
    sonda6.transform = Sondas[5].transform
    map.transform = cubo.transform

if __name__ == '__main__':

    pyglet.clock.schedule(update, controller)
    pyglet.app.run()