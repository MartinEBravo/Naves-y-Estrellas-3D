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
    "aliado": getAssetPath("enemy.obj"),
    "meteorito": getAssetPath("meteorite.obj")
}

class Controller(pyglet.window.Window):
    def __init__(self, width, height, title="Naves y Estrellas 3D"):
        super().__init__(width, height, title)
        self.total_time = 0.0
        self.pipeline = ls.SimpleGouraudShaderProgram()
        self.label = pyglet.text.Label("Meteoritos reiniciados: 0",
                                       font_name="Arial",
                                       font_size=12,
                                       x=10, y=height - 30,
                                       anchor_x="left", anchor_y="top")


class Camera:
    def __init__(self):
        self.at = np.array([0.0, 0.0, 0.0])
        self.eye = np.array([0.0, 0.0, 0.0])
        self.up = np.array([1.0, 0, 0])
        self.projection = tr.ortho(-8, 8, -8, 8, 0.1, 100)

    def update(self, dt):
        pos = np.array([0, 0, 0])
        self.at = pos
        self.eye = pos + np.array([-1, 3, 0])

controller = Controller(800, 800)
camera = Camera()

class Nave:
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z

        self.X_speed = 0.0
        self.Y_speed = 0.0
        self.Z_speed = 0.0
        self.rotation = 0.0
        self.phi = 0.0 

        # Vidas de la nave
        self.vidas = 3

        self.transform = tr.identity()

        self.collision = False  # Variable para controlar la colisión
        self.collision_point = None  # Punto de colisión

    def update(self, dt):
        if not self.collision:  # Solo se actualiza si no ha habido colisión
            self.X += dt * self.X_speed
            self.Y += dt * self.Y_speed
            self.Z += dt * self.Z_speed

            self.transform = tr.matmul([
                tr.translate(self.X, self.Y, self.Z),
                tr.uniformScale(0.2),
                tr.rotationY(-np.pi / 2),
                tr.rotationZ(self.rotation)
            ])
        else:
            self.phi += 1 * dt
            self.transform = tr.matmul([
                tr.translate(self.collision_point[0], self.collision_point[1], self.collision_point[2]),  # Usar el punto de colisión
                tr.uniformScale(0.2),
                tr.rotationY(self.phi),
                tr.rotationZ(self.phi)
            ])
        self.X = max(min(self.X, 6), -6)
        self.Z = max(min(self.Z, 7), -7)

    def set_collision_point(self, point):
        self.collision_point = point
        self.collision = True

class Aliado:
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z

        self.Y_speed = random.uniform(1,5)
        self.tam = random.uniform(0.1,0.15)

        self.transform = tr.identity()


    def update(self, dt):
        self.X += dt * self.Y_speed

        self.transform = tr.matmul([
            tr.translate(self.X, self.Y, self.Z),
            tr.uniformScale(self.tam),
            tr.rotationY(-np.pi /2),
        ])

        if self.X > 15:
            self.X -= 30
            self.Z = random.uniform(-8,8)
            self.vy = random.uniform(1, 5)  # Asegúrate de que las velocidades no sean cero
            self.tam = random.uniform(0.1, 0.15)
        #self.X = max(min(self.X, 8), -8)
        #
        # self.Z = max(min(self.Z, 8), -8)

class Estrella:
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z
        self.vy = random.uniform(-3, -1)  # Asegúrate de que las velocidades no sean cero
        self.tam = random.uniform(0.03, 0.06)

        self.transform = tr.identity()

    def update(self, dt):
        self.X += self.vy * dt
        self.transform = tr.matmul([
            tr.translate(self.X, self.Y, self.Z),
            tr.uniformScale(self.tam)
        ])
        if self.X < -15:
            self.X += 30
            self.Z = random.uniform(-15,15)
            self.vy = random.uniform(-3, -1)  # Asegúrate de que las velocidades no sean cero
            self.tam = random.uniform(0.03, 0.06)


# Se define el Meteorito
class Meteoro:
    def __init__(self,x,y,z):
        self.X = x
        self.Y = y
        self.Z = z
        self.phi = np.pi
        self.term = random.randint(-5,5)

        self.vx = random.uniform(-1,1)
        self.vy = random.uniform(-10, -5)  # Asegúrate de que las velocidades no sean cero
        self.tam = random.uniform(0.01, 0.02)

        self.transform = tr.identity()

        
    def update(self, dt):
        self.X += self.vy * dt
        self.Z += self.vx * dt
        self.phi += self.term * np.pi/10 * dt
        self.transform = tr.matmul([tr.translate(self.X, self.Y, self.Z),
                                    tr.rotationY(self.phi),
                                    tr.rotationZ(self.phi),
                                    tr.uniformScale(self.tam)])
        if self.X < -15 or abs(self.Z) > 10:
            self.X += 30
            self.Z = random.uniform(-15,15)
            self.vx = random.uniform(-1,1)
            self.vy = random.uniform(-10, -5)  # Asegúrate de que las velocidades no sean cero
            self.tam = random.uniform(0.01, 0.02)
            global count
            count += 1
count = 0            

gpuNaveLider = createGPUShape(controller.pipeline, read_OBJ(ASSETS["nave"], (0.9, 0.9, 0.9)))
Naves = [Nave(0, 0, 0)]
nave = sg.SceneGraphNode("nave")
nave.childs += [gpuNaveLider]
squad = sg.SceneGraphNode("squad")
squad.childs += [nave]

num_estrellas = 500 # Si va muy lageado disminuir cantidad de estrellas
gpuCuboEstrella = createGPUShape(controller.pipeline, bs.createColorNormalsCube(1, 1, 1))
estrellas = []
for i in range(num_estrellas):
    x = random.uniform(-15, 15)
    y = -5
    z = random.uniform(-15, 15)

    estrella = Estrella(x, y, z)
    
    estrellas.append(estrella)
    nodo_estrella = sg.SceneGraphNode("estrella" + str(i))
    nodo_estrella.transform = estrella.transform
    nodo_estrella.childs += [gpuCuboEstrella]
    squad.childs += [nodo_estrella]

num_meteoros = 50
gpuMeteoro = createGPUShape(controller.pipeline, read_OBJ(ASSETS["meteorito"], (0.2, 0.2, 0.2)))
meteoros = []
for i in range(num_meteoros):
    x = random.uniform(15, 30)
    y = 0
    z = random.uniform(-15, 15)

    meteoro = Meteoro(x, y, z)

    meteoros.append(meteoro)
    nodo_meteoro = sg.SceneGraphNode("meteoro" + str(i))
    nodo_meteoro.transform = meteoro.transform
    nodo_meteoro.childs += [gpuMeteoro]
    squad.childs += [nodo_meteoro]

num_naves = 5
gpuNave = createGPUShape(controller.pipeline, read_OBJ(ASSETS["nave"], (0.9, 0.9, 0.9)))
aliados = []
for i in range(num_naves):
    x = random.uniform(-8, 8)
    y = -4
    z = random.uniform(-8, 8)
    aliado = Aliado(x,y,z)
    aliados.append(aliado)
    nodo_aliado = sg.SceneGraphNode("aliado" + str(i))
    nodo_aliado.transform = aliado.transform
    nodo_aliado.childs += [gpuNave]
    squad.childs += [nodo_aliado]



glClearColor(0, 0, 0, 1.0)
glEnable(GL_DEPTH_TEST)
glUseProgram(controller.pipeline.shaderProgram)

glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Ka"), 0.5, 0.5, 0.5)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)
glUniform3f(glGetUniformLocation(controller.pipeline.shaderProgram, "lightPosition"), 10, 0, 0)
glUniform1ui(glGetUniformLocation(controller.pipeline.shaderProgram, "shininess"), 100)
glUniform1f(glGetUniformLocation(controller.pipeline.shaderProgram, "constantAttenuation"), 0.0001)
glUniform1f(glGetUniformLocation(controller.pipeline.shaderProgram, "linearAttenuation"), 0.03)
glUniform1f(glGetUniformLocation(controller.pipeline.shaderProgram, "quadraticAttenuation"), 0.01)


@controller.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.W:
        for nave in Naves:
            nave.X_speed = 5
    elif symbol == pyglet.window.key.S:
        for nave in Naves:
            nave.X_speed = -5
    elif symbol == pyglet.window.key.A:
        for nave in Naves:
            nave.Z_speed = -5
            nave.rotation = np.pi/4
    elif symbol == pyglet.window.key.D:
        for nave in Naves:
            nave.Z_speed = 5
            nave.rotation = -np.pi/4


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
            nave.rotation = 0
    elif symbol == pyglet.window.key.D:
        for nave in Naves:
            nave.Z_speed = 0
            nave.rotation = 0


music = pyglet.resource.media('musica.mp3')
music.play()


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
    glUniformMatrix4fv(glGetUniformLocation(controller.pipeline.shaderProgram, "projection"), 1, GL_TRUE,
                       camera.projection)
    glUniformMatrix4fv(glGetUniformLocation(controller.pipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    sg.drawSceneGraphNode(squad, controller.pipeline, "model")

def update(dt, controller):
    global count
    controller.total_time += dt

    for i, nave in enumerate(Naves):
        nave.update(dt)
        squad.childs[i].transform = nave.transform

    for i, estrella in enumerate(estrellas):
        estrella.update(dt)
        squad.childs[i + len(Naves)].transform = estrella.transform 

    for i, meteoro in enumerate(meteoros):
        meteoro.update(dt)
        squad.childs[i + len(Naves) + len(estrellas)].transform = meteoro.transform 
    
    for i, aliado in enumerate(aliados):
        aliado.update(dt)
        squad.childs[i + len(Naves) + len(estrellas) + len(meteoros)].transform = aliado.transform 

    camera.update(dt)

    # Colisión
    for nave in Naves:
        for meteoro in meteoros:
            if abs(nave.X - meteoro.X) < 0.5 and abs(nave.Z - meteoro.Z) < 0.5:
                nave.vidas -= 1
                # Si no ha habido colisión previa
                if not nave.collision and nave.vidas <= 0:
                    # Guarda el punto de colisión
                    nave.set_collision_point((nave.X, nave.Y, nave.Z)) 
                    # Calcula las velocidades en dirección opuesta
                    nave.X_speed = -nave.X_speed
                    nave.Z_speed = -nave.Z_speed
                meteoro.vx = -meteoro.vx
                meteoro.vy = -meteoro.vy

if __name__ == '__main__':
    pyglet.clock.schedule(update, controller)
    pyglet.app.run()
