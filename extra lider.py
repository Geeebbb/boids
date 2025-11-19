import vispy
vispy.use('pyglet')
import numpy as np
from numba import njit, prange
from vispy import app, scene
import time
import ffmpeg
import os
import shutil
from PIL import Image
import threading
import queue
import math

# --- ПАРАМЕТРЫ СИМУЛЯЦИИ ---
width, height = 2040, 1280
view_radius = 50.0
view_radius_sq = view_radius * view_radius
max_speed = 10.0
dt = 0.1
margin = 10.0  # Отступ от границ

# Параметры взаимодействия между классами агентов
# Структура: [alignment, cohesion, separation, wall_avoidance, noise]
params_array = np.array([
    [0.8, 0.6, 0.3, 0.4, 0.1],  # класс 0 → класс 0
    [0.1, 0.1, 0.9, 0.5, 0.2],  # класс 0 → класс 1
    [0.1, 0.1, 0.9, 0.5, 0.2],  # класс 1 → класс 0
    [0.4, 0.2, 0.7, 0.3, 0.15],  # класс 1 → класс 1
], dtype=np.float32)

leader_index = 0  # Индекс агента-лидера
leader_influence = 0.5  # Сила влияния лидера на остальных агентов


class BoidSimulation:
    """
    Класс симуляции поведения агентов (boids) с двумя классами.
    Использует пространственную сетку для оптимизации поиска соседей.
    """

    def __init__(self, n1=500, n2=500):
        self.num_agents = n1 + n2
        self.classes = np.zeros(self.num_agents, dtype=np.int32)
        self.classes[n1:] = 1

        self.positions = np.random.uniform([0, 0], [width, height], (self.num_agents, 2)).astype(np.float32)
        self.velocities = np.random.uniform(-1, 1, (self.num_agents, 2)).astype(np.float32)
        self.velocities = self.normalize_vectors(self.velocities) * max_speed

        self.cell_size = view_radius
        self.grid_width = math.ceil(width / self.cell_size)
        self.grid_height = math.ceil(height / self.cell_size)
        self.grid = np.full((self.grid_height, self.grid_width), -1, dtype=np.int32)
        self.next_agent = np.full(self.num_agents, -1, dtype=np.int32)

    def update_grid(self):
        self.grid.fill(-1)
        self.next_agent.fill(-1)
        for i in range(self.num_agents):
            x, y = self.positions[i]
            grid_x = min(int(x / self.cell_size), self.grid_width - 1)
            grid_y = min(int(y / self.cell_size), self.grid_height - 1)
            head = self.grid[grid_y, grid_x]
            self.grid[grid_y, grid_x] = i
            self.next_agent[i] = head

    @staticmethod
    @njit(fastmath=True)
    def normalize_vectors(vectors):
        mags = np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)
        return vectors / np.where(mags > 0, mags, 1).reshape(-1, 1)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def compute_interactions_optimized(positions, velocities, classes, params_arr,
                                       grid, next_agent, grid_width, grid_height,
                                       cell_size, width, height, margin,
                                       view_radius_sq, leader_index, leader_influence):
        n = positions.shape[0]
        acc = np.zeros_like(positions)
        leader_pos = positions[leader_index]

        for i in prange(n):
            pos_i = positions[i]
            class_i = classes[i]
            alignment = np.zeros(2)
            cohesion = np.zeros(2)
            separation = np.zeros(2)
            count = 0
            grid_x = min(int(pos_i[0] / cell_size), grid_width - 1)
            grid_y = min(int(pos_i[1] / cell_size), grid_height - 1)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    check_x = grid_x + dx
                    check_y = grid_y + dy
                    if 0 <= check_x < grid_width and 0 <= check_y < grid_height:
                        j = grid[check_y, check_x]
                        while j >= 0:
                            if i == j:
                                j = next_agent[j]
                                continue
                            class_j = classes[j]
                            p_idx = class_i * 2 + class_j
                            p = params_arr[p_idx]
                            dx_pos = positions[j, 0] - pos_i[0]
                            dy_pos = positions[j, 1] - pos_i[1]
                            dist_sq = dx_pos * dx_pos + dy_pos * dy_pos
                            if dist_sq < view_radius_sq and dist_sq > 1e-8:
                                dist = math.sqrt(dist_sq)
                                alignment += velocities[j] * p[0]
                                cohesion += np.array([dx_pos, dy_pos]) * p[1]
                                separation -= np.array([dx_pos, dy_pos]) / (dist_sq + 1e-8) * p[2]
                                count += 1
                            j = next_agent[j]
            pself = params_arr[class_i * 2 + class_i]
            wall = np.zeros(2)
            if pos_i[0] < margin:
                wall[0] = (margin - pos_i[0]) / margin
            elif pos_i[0] > width - margin:
                wall[0] = (width - margin - pos_i[0]) / margin
            if pos_i[1] < margin:
                wall[1] = (margin - pos_i[1]) / margin
            elif pos_i[1] > height - margin:
                wall[1] = (height - margin - pos_i[1]) / margin
            wall *= pself[3]
            noise = (np.random.rand(2) * 2 - 1) * pself[4]

            # Влияние лидера (кроме самого лидера)
            leader_force = np.zeros(2)
            if i != leader_index:
                vec_to_leader = leader_pos - pos_i
                dist_to_leader_sq = vec_to_leader[0] * vec_to_leader[0] + vec_to_leader[1] * vec_to_leader[1]
                if dist_to_leader_sq > 1e-8:
                    dist_to_leader = math.sqrt(dist_to_leader_sq)
                    leader_dir = vec_to_leader / dist_to_leader
                    leader_force = leader_dir * leader_influence

            if count > 0:
                alignment /= count
                cohesion /= count
                separation /= count

            acc[i] = alignment + cohesion + separation + wall + noise + leader_force
        return acc

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def update(positions, velocities, acc):
        n = positions.shape[0]
        for i in prange(n):
            velocities[i] += acc[i] * dt
            speed = np.sqrt(velocities[i, 0] ** 2 + velocities[i, 1] ** 2)
            if speed > max_speed:
                velocities[i] = velocities[i] / speed * max_speed
            positions[i] += velocities[i] * dt


            if positions[i, 0] < margin:
                positions[i, 0] = margin
                velocities[i, 0] = -velocities[i, 0] * 0.5
            elif positions[i, 0] > width - margin:
                positions[i, 0] = width - margin
                velocities[i, 0] = -velocities[i, 0] * 0.5

            if positions[i, 1] < margin:
                positions[i, 1] = margin
                velocities[i, 1] = -velocities[i, 1] * 0.5
            elif positions[i, 1] > height - margin:
                positions[i, 1] = height - margin
                velocities[i, 1] = -velocities[i, 1] * 0.5


class BoidCanvas(scene.SceneCanvas):
    def __init__(self, sim):
        super().__init__(keys='interactive', size=(width, height))
        self.unfreeze()
        self.sim = sim
        self.view = self.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(0, 0, width, height))
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.text = scene.visuals.Text('', color='white', pos=(10, 10),
                                       anchor_x='left', anchor_y='bottom',
                                       face='Arial',
                                       parent=self.scene)
        self.frame_count = 0
        self.last_time = time.time()
        self._fps = 0.0
        self.timer = app.Timer(1 / 60, connect=self.on_timer, start=True)
        self.recording = False
        self.frame_index = 0
        self.frames_dir = 'frames'
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir, exist_ok=True)
        self.frame_queue = queue.Queue(maxsize=5000)
        self.num_workers = 5
        self.save_threads = []
        self.stop_event = threading.Event()
        for _ in range(self.num_workers):
            t = threading.Thread(target=self.save_worker, daemon=True)
            t.start()
            self.save_threads.append(t)
        self.freeze()
        self.show()

    def save_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=0.1)
                if item is None:
                    self.frame_queue.task_done()
                    break
                img_array, filename = item
                img = Image.fromarray(img_array)
                img.save(filename)
                self.frame_queue.task_done()
            except queue.Empty:
                continue

    def on_timer(self, event):
        self.sim.update_grid()
        acc = self.sim.compute_interactions_optimized(
            self.sim.positions,
            self.sim.velocities,
            self.sim.classes,
            params_array,
            self.sim.grid,
            self.sim.next_agent,
            self.sim.grid_width,
            self.sim.grid_height,
            self.sim.cell_size,
            width,
            height,
            margin,
            view_radius_sq,
            leader_index,
            leader_influence
        )
        BoidSimulation.update(self.sim.positions, self.sim.velocities, acc)

        colors = np.zeros((self.sim.num_agents, 4), dtype=np.float32)
        colors[self.sim.classes == 0] = [1, 0, 0, 1]
        colors[self.sim.classes == 1] = [0, 1, 0, 1]
        colors[leader_index] = [0, 0, 1, 1]

        self.scatter.set_data(self.sim.positions, face_color=colors, size=5)

        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self._fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now

        self.text.text = (f'Agents: {self.sim.num_agents}    '
                          f'FPS: {self._fps:.1f}\n'
                          f'Params:\n{params_array}\n'
                          f'{"Recording..." if self.recording else "Press R to record"}')

        if self.recording:
            try:
                img_array = self.render()
                filename = os.path.join(self.frames_dir, f'frame_{self.frame_index:05d}.png')
                self.frame_queue.put_nowait((img_array, filename))
                self.frame_index += 1
            except queue.Full:
                print("Frame queue is full, skipping frame")
            except Exception as e:
                print(f"Error capturing frame: {e}")

        self.update()

    def on_key_press(self, event):
        if event.text.lower() == 'r':
            self.recording = not self.recording
            if self.recording:
                print("Started recording")
                self.frame_index = 0
                if os.path.exists(self.frames_dir):
                    shutil.rmtree(self.frames_dir)
                os.makedirs(self.frames_dir, exist_ok=True)
            else:
                print("Stopped recording, assembling video...")
                self.frame_queue.join()
                self.stop_event.set()
                for _ in self.save_threads:
                    self.frame_queue.put(None)
                for t in self.save_threads:
                    t.join()
                self.save_threads.clear()
                self.stop_event.clear()
                threading.Thread(target=self.create_video, daemon=True).start()

    def create_video(self):
        try:
            input_pattern = os.path.join(self.frames_dir, 'frame_%05d.png')
            output_file = 'boids_simulation.mp4'

            if not os.path.exists(os.path.join(self.frames_dir, 'frame_00000.png')):
                print("No frames to create video")
                return

            (
                ffmpeg
                .input(input_pattern, framerate=60)
                .output(output_file, vcodec='libx264', pix_fmt='yuv420p', crf=18)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            print(f"Video successfully saved as {output_file}")
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode('utf8')}")
        except Exception as e:
            print(f"Error creating video: {e}")


if __name__ == '__main__':
    sim = BoidSimulation(n1=1500, n2=1500)
    canvas = BoidCanvas(sim)
    app.run()