import vispy
vispy.use('pyglet')
import numpy as np
from numba import njit, prange
from vispy import app, scene
import time, ffmpeg, os, shutil, threading, queue, math
from PIL import Image

# --- ПАРАМЕТРЫ СИМУЛЯЦИИ ---
width, height = 2040, 1280
view_radius = 50.0
view_radius_sq = view_radius * view_radius
max_speed = 10.0
dt = 0.1
margin = 10.0

params_array = np.array([
    [0.8, 0.6, 0.3, 0.4, 0.1],
    [0.1, 0.1, 0.9, 0.5, 0.2],
    [0.1, 0.1, 0.9, 0.5, 0.2],
    [0.4, 0.2, 0.7, 0.3, 0.15],
], dtype=np.float32)


class BoidSimulation:
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
        self.obstacles = [
            (800.0, 400.0, 100.0),
            (1400.0, 900.0, 120.0),
            (600.0, 1000.0, 80.0)
        ]

    def update_grid(self):
        self.grid.fill(-1)
        self.next_agent.fill(-1)
        for i in range(self.num_agents):
            x, y = self.positions[i]
            gx = min(int(x / self.cell_size), self.grid_width - 1)
            gy = min(int(y / self.cell_size), self.grid_height - 1)
            head = self.grid[gy, gx]
            self.grid[gy, gx] = i
            self.next_agent[i] = head

    @staticmethod
    @njit(fastmath=True)
    def normalize_vectors(v):
        mags = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
        return v / np.where(mags > 0, mags, 1).reshape(-1, 1)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def compute_interactions_optimized(positions, velocities, classes, params_arr,
                                       grid, next_agent, grid_width, grid_height,
                                       cell_size, width, height, margin,
                                       view_radius_sq, obstacles):
        n = positions.shape[0]
        acc = np.zeros_like(positions)
        for i in prange(n):
            pos_i = positions[i]
            class_i = classes[i]
            alignment = np.zeros(2)
            cohesion = np.zeros(2)
            separation = np.zeros(2)
            count = 0
            gx = min(int(pos_i[0] / cell_size), grid_width - 1)
            gy = min(int(pos_i[1] / cell_size), grid_height - 1)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    cx, cy = gx + dx, gy + dy
                    if 0 <= cx < grid_width and 0 <= cy < grid_height:
                        j = grid[cy, cx]
                        while j >= 0:
                            if j != i:
                                p_idx = class_i * 2 + classes[j]
                                p = params_arr[p_idx]
                                dxj = positions[j, 0] - pos_i[0]
                                dyj = positions[j, 1] - pos_i[1]
                                dsq = dxj * dxj + dyj * dyj
                                if dsq < view_radius_sq and dsq > 1e-8:
                                    dist = math.sqrt(dsq)
                                    alignment += velocities[j] * p[0]
                                    cohesion += np.array([dxj, dyj]) * p[1]
                                    separation -= np.array([dxj, dyj]) / (dsq + 1e-8) * p[2]
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

            avoidance = np.zeros(2)
            for k in range(len(obstacles)):
                ox, oy, r = obstacles[k]
                dxo = pos_i[0] - ox
                dyo = pos_i[1] - oy
                dsqo = dxo * dxo + dyo * dyo
                rad = r + 100.0  # радиус избегания
                if dsqo < rad * rad and dsqo > 1e-8:
                    dist = math.sqrt(dsqo)
                    force = (rad - dist) / rad
                    avoidance += (np.array([dxo, dyo]) / dist) * force * 10.0  # усиление силы

            if count > 0:
                alignment /= count
                cohesion /= count
                separation /= count

            acc[i] = alignment + cohesion + separation + wall + noise + avoidance
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


class BoidCanvas(scene.SceneCanvas):
    def __init__(self, sim):
        super().__init__(keys='interactive', size=(width, height))
        self.unfreeze()
        self.sim = sim
        self.view = self.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(0, 0, width, height))
        self.scatter = scene.visuals.Markers(parent=self.view.scene)


        self.text = scene.visuals.Text(
            '',
            color='white',
            pos=(10, 10),
            anchor_x='left',
            anchor_y='bottom',
            font_size=12,
            face='Arial',
            parent=self.scene
        )

        # препятствий
        self.obstacle_visuals = []
        for x, y, r in sim.obstacles:
            circle = scene.visuals.Ellipse(
                center=(x, y),
                radius=(r, r),
                color=(0.5, 0.5, 0.5, 0.6),
                border_color='white',
                parent=self.view.scene
            )
            self.obstacle_visuals.append(circle)

        self.frame_count = 0
        self.last_time = time.time()
        self._fps = 0
        self.timer = app.Timer(1 / 60, connect=self.on_timer, start=True)

        #Видео запись
        self.recording = False
        self.frame_index = 0
        self.frames_dir = 'frames'
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir)

        self.frame_queue = queue.Queue(maxsize=5000)
        self.num_workers = 5
        self.save_threads = []
        self.stop_event = threading.Event()
        self.start_save_threads()

        self.freeze()
        self.show()

    def start_save_threads(self):
        self.stop_event.clear()
        self.save_threads = []
        for _ in range(self.num_workers):
            t = threading.Thread(target=self.save_worker, daemon=True)
            t.start()
            self.save_threads.append(t)

    def save_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=0.1)
                if item is None:
                    self.frame_queue.task_done()
                    break
                img_arr, fname = item
                Image.fromarray(img_arr).save(fname)
                self.frame_queue.task_done()
            except queue.Empty:
                pass

    def on_timer(self, event):
        self.sim.update_grid()
        acc = BoidSimulation.compute_interactions_optimized(
            self.sim.positions, self.sim.velocities, self.sim.classes,
            params_array,
            self.sim.grid, self.sim.next_agent,
            self.sim.grid_width, self.sim.grid_height,
            self.sim.cell_size, width, height, margin,
            view_radius_sq,
            np.array(self.sim.obstacles, dtype=np.float32)
        )
        BoidSimulation.update(self.sim.positions, self.sim.velocities, acc)

        colors = np.zeros((self.sim.num_agents, 4), dtype=np.float32)
        colors[self.sim.classes == 0] = [1, 0, 0, 1]
        colors[self.sim.classes == 1] = [0, 1, 0, 1]
        self.scatter.set_data(self.sim.positions, face_color=colors, size=5)

        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self._fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now

        self.text.text = (
            f'Agents: {self.sim.num_agents}   FPS: {self._fps:.1f}\n'
            f'Params:\n{params_array}\n'
            f'{"RECORDING..." if self.recording else "Press R to record"}'
        )

        self.update()

        if self.recording:
            img_arr = self.render()
            fname = os.path.join(self.frames_dir, f'frame_{self.frame_index:05d}.png')
            try:
                self.frame_queue.put_nowait((img_arr, fname))
                self.frame_index += 1
            except queue.Full:
                print("Frame queue full!")

    def on_key_press(self, event):
        if event.text.lower() == 'r':
            self.recording = not self.recording
            if self.recording:
                print("Recording started")
                if os.path.exists(self.frames_dir):
                    shutil.rmtree(self.frames_dir)
                os.makedirs(self.frames_dir)
                self.frame_index = 0
                if not any(t.is_alive() for t in self.save_threads):
                    self.start_save_threads()
            else:
                print("Recording stopped, encoding...")
                threading.Thread(target=self.create_video, daemon=True).start()

    def create_video(self):
        self.frame_queue.join()
        self.stop_event.set()
        for _ in range(self.num_workers):
            self.frame_queue.put(None)
        for t in self.save_threads:
            t.join()
        try:
            (
                ffmpeg
                .input(os.path.join(self.frames_dir, 'frame_%05d.png'), framerate=60)
                .output('boids_simulation.mp4',
                        pix_fmt='yuv420p',
                        vcodec='libx264',
                        crf=23,
                        preset='fast')
                .overwrite_output()
                .run(cmd=['ffmpeg', '-y'])
            )
            print("Video saved successfully")
        except ffmpeg.Error as e:
            print(f"Error encoding video: {e.stderr.decode('utf8')}")
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir)
        self.frame_index = 0
        self.stop_event.clear()
        self.start_save_threads()
        print("Ready for new recording.")
def main():
    sim = BoidSimulation(n1=2500, n2=2500)
    canvas = BoidCanvas(sim)
    app.run()
if __name__ == "__main__":
    main()