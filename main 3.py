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
view_radius_sq = view_radius * view_radius  # Предварительное вычисление квадрата радиуса для скорости
max_speed = 10.0
dt = 0.1
margin = 10.0  # Отступ от границ для правила стен

# Параметры взаимодействия между классами агентов
# Структура: [alignment, cohesion, separation, wall_avoidance, noise]
params_array = np.array([
    [0.8,  0.6, 0.3, 0.4, 0.1],  # класс 0 → класс 0: выравнивание и сплочение, умеренное отталкивание
    [0.1,  0.1, 0.9, 0.5, 0.2],  # класс 0 → класс 1: почти не ориентируются, но сильно отталкиваются
    [0.1,  0.1, 0.9, 0.5, 0.2],  # класс 1 → класс 0: симметрично — отталкиваются
    [0.4,  0.2, 0.7, 0.3, 0.15], # класс 1 → класс 1: слабое выравнивание, отталкивание преобладает
], dtype=np.float32)



class BoidSimulation:
    """
    Класс симуляции поведения агентов (boids) с двумя классами.
    Использует пространственную сетку для оптимизации поиска соседей.
    """

    def __init__(self, n1=500, n2=500):
        """
        Инициализация симуляции.

        Parameters:
            n1 (int): количество агентов класса 0
            n2 (int): количество агентов класса 1
        """
        self.num_agents = n1 + n2
        self.classes = np.zeros(self.num_agents, dtype=np.int32)
        self.classes[n1:] = 1  # Агенты с индекса n1 до конца относятся к классу 1

        # Случайное начальное положение и скорость агентов
        self.positions = np.random.uniform([0, 0], [width, height], (self.num_agents, 2)).astype(np.float32)
        self.velocities = np.random.uniform(-1, 1, (self.num_agents, 2)).astype(np.float32)
        self.velocities = self.normalize_vectors(self.velocities) * max_speed

        # Параметры для пространственной сетки
        self.cell_size = view_radius  # Размер ячейки сетки равен радиусу видимости
        self.grid_width = math.ceil(width / self.cell_size)
        self.grid_height = math.ceil(height / self.cell_size)
        self.grid = np.full((self.grid_height, self.grid_width), -1, dtype=np.int32)
        self.next_agent = np.full(self.num_agents, -1, dtype=np.int32)

    def update_grid(self):
        """
        Обновляет пространственную сетку, размещая агентов по ячейкам.
        Создает связанные списки агентов для каждой ячейки.
        """
        self.grid.fill(-1)  # Сброс сетки
        self.next_agent.fill(-1)  # Сброс ссылок на следующих агентов

        for i in range(self.num_agents):
            x, y = self.positions[i]
            # Вычисление индекса ячейки, обрезая до границ сетки
            grid_x = min(int(x / self.cell_size), self.grid_width - 1)
            grid_y = min(int(y / self.cell_size), self.grid_height - 1)

            # Добавление агента в начало связанного списка ячейки
            head = self.grid[grid_y, grid_x]
            self.grid[grid_y, grid_x] = i
            self.next_agent[i] = head

    @staticmethod
    @njit(fastmath=True)
    def normalize_vectors(vectors):
        """
        Нормализует массив 2D-векторов.
        Parameters:
            vectors (np.ndarray): массив формы (N, 2)
        Returns:
            np.ndarray: нормализованные векторы
        """
        mags = np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)
        return vectors / np.where(mags > 0, mags, 1).reshape(-1, 1)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def compute_interactions_optimized(positions, velocities, classes, params_arr,
                                       grid, next_agent, grid_width, grid_height,
                                       cell_size, width, height, margin,
                                       view_radius_sq):
        """
        Вычисляет результирующее ускорение для каждого агента на основе взаимодействий,
        используя оптимизацию с пространственной сеткой.

        Parameters:
            positions (np.ndarray): текущие позиции агентов (N, 2)
            velocities (np.ndarray): текущие скорости агентов (N, 2)
            classes (np.ndarray): классы агентов (N,)
            params_arr (np.ndarray): массив параметров взаимодействий (4, 5)
            grid (np.ndarray): пространственная сетка (H, W)
            next_agent (np.ndarray): массив ссылок для связанных списков в сетке
            grid_width (int): ширина сетки
            grid_height (int): высота сетки
            cell_size (float): размер одной ячейки сетки
            width (float): ширина симуляционного пространства
            height (float): высота симуляционного пространства
            margin (float): отступ от границ
            view_radius_sq (float): квадрат радиуса видимости

        Returns:
            np.ndarray: массив ускорений для каждого агента (N, 2)
        """
        n = positions.shape[0]
        acc = np.zeros_like(positions)  # Итоговые ускорения
        for i in prange(n):  # Параллельный цикл по агентам
            pos_i = positions[i]
            class_i = classes[i]
            alignment = np.zeros(2)
            cohesion = np.zeros(2)
            separation = np.zeros(2)
            count = 0  # Количество соседей в радиусе видимости
            # Определяем ячейку текущего агента
            grid_x = min(int(pos_i[0] / cell_size), grid_width - 1)
            grid_y = min(int(pos_i[1] / cell_size), grid_height - 1)
            # Проверяем текущую и 8 соседних ячеек
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    check_x = grid_x + dx
                    check_y = grid_y + dy
                    # Проверяем, что соседняя ячейка находится в границах сетки
                    if 0 <= check_x < grid_width and 0 <= check_y < grid_height:
                        j = grid[check_y, check_x]
                        while j >= 0:
                            if i == j:  # Исключаем взаимодействие агента с самим собой
                                j = next_agent[j]
                                continue
                            class_j = classes[j]
                            # Определяем индекс параметров взаимодействия между классами i и j
                            p_idx = class_i * 2 + class_j
                            p = params_arr[p_idx]  # Извлекаем параметры
                            dx_pos = positions[j, 0] - pos_i[0]
                            dy_pos = positions[j, 1] - pos_i[1]
                            dist_sq = dx_pos * dx_pos + dy_pos * dy_pos  # Квадрат расстояния
                            # Проверяем, находится ли агент j в радиусе видимости агента i
                            if dist_sq < view_radius_sq and dist_sq > 1e-8:
                                dist = math.sqrt(dist_sq)  # Извлекаем корень только если агент в радиусе
                                # Применяем правила Boids с соответствующими весами
                                alignment += velocities[j] * p[0]
                                cohesion += np.array([dx_pos, dy_pos]) * p[1]
                                separation -= np.array([dx_pos, dy_pos]) / (dist_sq + 1e-8) * p[2]
                                count += 1
                            j = next_agent[j]  # Переходим к следующему агенту в связанном списке
            # Взаимодействие с границами
            pself = params_arr[
                class_i * 2 + class_i]
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
            noise = (np.random.rand(2) * 2 - 1) * pself[4]  # noise_weight
            if count > 0:
                alignment /= count
                cohesion /= count
                separation /= count
            # Суммируем все силы для получения итогового ускорения
            acc[i] = alignment + cohesion + separation + wall + noise
        return acc

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def update(positions, velocities, acc):
        """
        Обновляет состояния агентов на основе ускорений.

        Parameters:
            positions (np.ndarray): текущие позиции (N, 2)
            velocities (np.ndarray): текущие скорости (N, 2)
            acc (np.ndarray): ускорения (N, 2)
        """
        n = positions.shape[0]
        for i in prange(n):
            velocities[i] += acc[i] * dt
            # Ограничение максимальной скорости
            speed = np.sqrt(velocities[i, 0] ** 2 + velocities[i, 1] ** 2)
            if speed > max_speed:
                velocities[i] = velocities[i] / speed * max_speed

            positions[i] += velocities[i] * dt

            # щас агенты отталкивались от стен
            # positions[i, 0] %= width
            # positions[i, 1] %= height


class BoidCanvas(scene.SceneCanvas):
    """
    Визуализация и управление симуляцией boids с помощью библиотеки Vispy.
    """

    def __init__(self, sim):
        """
        Инициализация канвы.

        Parameters:
            sim (BoidSimulation): объект симуляции
        """
        super().__init__(keys='interactive', size=(width, height))
        self.unfreeze()
        self.sim = sim
        self.view = self.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(0, 0, width, height))  # Камера
        self.scatter = scene.visuals.Markers(parent=self.view.scene)  # Визуализация агентов как маркеров
        # Текст на видео
        self.text = scene.visuals.Text('', color='white', pos=(10, 10),
                                       anchor_x='left', anchor_y='bottom',
                                       face='Arial',
                                       parent=self.scene)

        self.frame_count = 0
        self.last_time = time.time()
        self._fps = 0.0
        # Таймер Vispy, вызывающий on_timer 60 раз в секунду
        self.timer = app.Timer(1 / 60, connect=self.on_timer, start=True)
        self.recording = False
        self.frame_index = 0
        self.frames_dir = 'frames'
        #создание директории для кадров
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir)

        # Очередь для передачи кадров потокам сохранения
        self.frame_queue = queue.Queue(maxsize=5000)  # Увеличиваем размер очереди
        self.num_workers = 5  # Количество потоков для сохранения кадров
        self.save_threads = []
        self.stop_event = threading.Event()  # Событие для сигнализации потокам о завершении работы

        # Запуск потоков сохранения
        for _ in range(self.num_workers):
            t = threading.Thread(target=self.save_worker, daemon=True)
            t.start()
            self.save_threads.append(t)

        self.freeze()  # Запрещаем изменение структуры сцены для оптимизации
        self.show()  # Показываем окно

    def save_worker(self):
        """
        Фоновый поток для сохранения кадров изображения на диск.
        Используется при записи видео для предотвращения блокировки основного потока.
        """
        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=0.1)
                if item is None:  # Сигнал о завершении работы потока
                    self.frame_queue.task_done()
                    break
                img_array, filename = item
                img = Image.fromarray(img_array)
                img.save(filename)
                self.frame_queue.task_done()  # Сообщаем очереди, что задача выполнена
            except queue.Empty:
                continue  # Если очередь пуста, продолжаем ждать

    def on_timer(self, event):
        """
        Основной цикл обновления симуляции и визуализации. Вызывается 60 раз в секунду.
        Обновляет симуляцию, отображает позиции агентов и сохраняет кадры при записи.
        """
        #Обновление пространственной сетки
        self.sim.update_grid()
        # Вычисление взаимодействий с использованием оптимизированной функции
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
            view_radius_sq
        )

        # Обновление позиций и скоростей агентов
        BoidSimulation.update(self.sim.positions, self.sim.velocities, acc)

        # Обновление цветов для визуализации (красный для класса 0, зеленый для класса 1)
        colors = np.zeros((self.sim.num_agents, 4), dtype=np.float32)
        colors[self.sim.classes == 0] = [1, 0, 0, 1]  # Красный
        colors[self.sim.classes == 1] = [0, 1, 0, 1]  # Зеленый

        self.scatter.set_data(self.sim.positions, face_color=colors, size=5)

        # Расчет и отображение FPS
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:  # Обновляем FPS каждую секунду
            self._fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now

        #Обновление текста на экране
        self.text.text = (f'Agents: {self.sim.num_agents}    '
                          f'FPS: {self._fps:.1f}\n'
                          f'Params:\n{params_array}\n'
                          f'{"Recording..." if self.recording else "Press R to record"}')

        #Обновление канвы Vispy
        self.update()

        # Запись кадра, если включена запись
        if self.recording:
            img_array = self.render()  # Получаем пиксели текущего кадра
            filename = os.path.join(self.frames_dir, f'frame_{self.frame_index:05d}.png')  # Форматируем имя файла
            try:
                self.frame_queue.put_nowait((img_array, filename))  # Добавляем кадр в очередь
                self.frame_index += 1
            except queue.Full:
                print("Frame queue full! Dropping frame.")  # Предупреждение, если очередь переполнена

    def on_key_press(self, event):
        """
        Обработчик нажатий клавиш.
        'R' - начать/остановить запись видео.

        Parameters:
            event (KeyEvent): событие клавиши
        """
        if event.text.lower() == 'r':
            self.recording = not self.recording  # Переключение состояния записи
            if self.recording:
                print("Recording started")
                # При старте записи, очищаем предыдущие кадры и сбрасываем индекс
                if os.path.exists(self.frames_dir):
                    shutil.rmtree(self.frames_dir)
                os.makedirs(self.frames_dir)
                self.frame_index = 0
                self.stop_event.clear()

                # Перезапускаем потоки сохранения при каждом старте записи, чтобы убедиться, что они готовы к новой серии кадров
                for t in self.save_threads:
                    if t.is_alive():
                        self.frame_queue.put(None)  # Сигнал на завершение
                for t in self.save_threads:
                    if t.is_alive():
                        t.join()
                self.save_threads = []  # Очищаем список потоков

                # Запускаем новые потоки для новой сессии записи
                for _ in range(self.num_workers):
                    t = threading.Thread(target=self.save_worker, daemon=True)
                    t.start()
                    self.save_threads.append(t)
            else:
                print("Recording stopped. Starting video encoding in background...")
                # При остановке записи запускаем процесс кодирования видео в отдельном потоке
                threading.Thread(target=self.create_video, daemon=True).start()

    def create_video(self):
        """
        Собирает PNG-кадры в mp4-видео с использованием ffmpeg.
        Выполняется в отдельном потоке, чтобы не блокировать GUI.
        """
        print("Starting video encoding...")

        # Ждем, пока все кадры из очереди будут сохранены на диск
        self.frame_queue.join()
        # Сигнализируем всем потокам сохранения о завершении работы
        self.stop_event.set()
        for _ in range(self.num_workers):
            self.frame_queue.put(None)
        for t in self.save_threads:
            t.join()

        input_pattern = os.path.join(self.frames_dir, 'frame_%05d.png')
        output_file = 'boids_simulation.mp4'

        try:
            (
                ffmpeg
                .input(input_pattern, framerate=60)  # Указываем 60 кадров в секунду
                .output(output_file, pix_fmt='yuv420p', vcodec='libx264', crf=23)  # Кодек H.264, CRF для качества
                .overwrite_output()  # Перезаписывать, если файл существует
                .run(cmd=['ffmpeg', '-y'])  # Явно указываем команду ffmpeg и флаг перезаписи
            )
            print(f"Video saved to {output_file}")
        except Exception as e:
            print(f"FFmpeg error: {e}")
            print("Please ensure FFmpeg is installed and accessible in your system's PATH.")

        # Очистка директории с кадрами после создания видео и подготовка к новой записи
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir)
        self.frame_index = 0
        self.recording = False
        self.stop_event.clear()  # Сброс события для следующей записи
        print("Ready for new recording.")
def main():
    """
    Запускает симуляцию и визуализацию.
    """
    # Создаем симуляцию с 2500 агентами класса 0 и 2500 агентами класса 1 (всего 5000)
    sim = BoidSimulation(n1=2500, n2=2500)
    canvas = BoidCanvas(sim)
    app.run()  # Запуск цикла событий Vispy
if __name__ == "__main__":
    main()