import cv2
import threading
import time
from flask import Flask, Response, render_template_string, request

class WebGUI:
    """Веб-интерфейс с поддержкой нескольких окон (видеопотоков) и трекбаров."""

    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._images = {}           # имя окна -> текущее изображение (numpy)
        self._trackbars = {}        # имя трекбара -> (min, max, callback)
        self._trackbar_values = {}  # имя трекбара -> текущее значение
        self._lock = threading.Lock()
        self._running = True
        self._setup_routes()
        # Запуск сервера в фоновом потоке
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        time.sleep(0.5)  # даём время на запуск

    def _run_server(self):
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True)

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            # HTML-шаблон с динамическим добавлением окон
            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>WebGUI - Multiple Windows</title>
                <style>
                    body { font-family: Arial; margin: 20px; }
                    .windows { display: flex; flex-wrap: wrap; gap: 20px; }
                    .window { flex: 1 1 45%; border: 1px solid #ccc; padding: 10px; text-align: center; }
                    .window h3 { margin-top: 0; }
                    .window img { width: 100%; height: auto; background: #f0f0f0; }
                    .trackbars { margin-top: 30px; padding: 10px; background: #f8f8f8; border: 1px solid #ddd; }
                    .slider { margin: 10px 0; }
                    label { display: inline-block; width: 120px; }
                    input[type=range] { width: 300px; }
                    .value { display: inline-block; width: 40px; text-align: right; }
                </style>
            </head>
            <body>
                <h1>WebGUI - Multiple Image Windows</h1>
                <div id="windows" class="windows"></div>
                <div id="trackbars" class="trackbars">
                    <h2>Trackbars</h2>
                    <div id="trackbars-list"></div>
                </div>
                <script>
                    // Получение списка окон и трекбаров с сервера
                    function fetchWindows() {
                        fetch('/windows').then(r => r.json()).then(data => {
                            let container = document.getElementById('windows');
                            let currentWindows = new Set();
                            for (let win of data) {
                                currentWindows.add(win);
                                if (!document.getElementById(`win_${win}`)) {
                                    let div = document.createElement('div');
                                    div.className = 'window';
                                    div.id = `win_${win}`;
                                    div.innerHTML = `<h3>${win}</h3>
                                                     <img src="/video_feed/${win}" id="img_${win}" style="width:100%">`;
                                    container.appendChild(div);
                                }
                            }
                            // Удалить окна, которых больше нет
                            for (let winDiv of container.children) {
                                let winName = winDiv.id.replace('win_', '');
                                if (!currentWindows.has(winName)) {
                                    container.removeChild(winDiv);
                                }
                            }
                        });
                    }

                    function fetchTrackbars() {
                        fetch('/trackbars').then(r => r.json()).then(data => {
                            let container = document.getElementById('trackbars-list');
                            let currentBars = new Set();
                            for (let name in data) {
                                currentBars.add(name);
                                if (!document.getElementById(`tb_${name}`)) {
                                    let div = document.createElement('div');
                                    div.className = 'slider';
                                    div.id = `tb_${name}`;
                                    let tb = data[name];
                                    div.innerHTML = `
                                        <label>${name}:</label>
                                        <input type="range" min="${tb.min}" max="${tb.max}" value="${tb.value}"
                                               oninput="updateTrackbar('${name}', this.value); document.getElementById('${name}_val').innerText=this.value">
                                        <span id="${name}_val" class="value">${tb.value}</span>
                                    `;
                                    container.appendChild(div);
                                } else {
                                    // обновить значение, если изменилось на сервере
                                    let input = document.querySelector(`#tb_${name} input`);
                                    let span = document.getElementById(`${name}_val`);
                                    if (input && data[name].value !== parseInt(input.value)) {
                                        input.value = data[name].value;
                                        span.innerText = data[name].value;
                                    }
                                }
                            }
                            // Удалить трекбары, которых больше нет
                            for (let tbDiv of container.children) {
                                let tbName = tbDiv.id.replace('tb_', '');
                                if (!currentBars.has(tbName)) {
                                    container.removeChild(tbDiv);
                                }
                            }
                        });
                    }

                    function updateTrackbar(name, value) {
                        fetch('/trackbar', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({name: name, value: parseInt(value)})
                        });
                    }

                    setInterval(fetchWindows, 200);
                    setInterval(fetchTrackbars, 200);
                </script>
            </body>
            </html>
            '''
            return render_template_string(html)

        @self.app.route('/windows')
        def list_windows():
            with self._lock:
                return list(self._images.keys())

        @self.app.route('/video_feed/<winname>')
        def video_feed(winname):
            def generate():
                while self._running:
                    with self._lock:
                        img = self._images.get(winname)
                    if img is not None:
                        _, jpeg = cv2.imencode('.jpg', img)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.05)
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/trackbars')
        def get_trackbars():
            with self._lock:
                data = {}
                for name, (minv, maxv, _) in self._trackbars.items():
                    data[name] = {'min': minv, 'max': maxv, 'value': self._trackbar_values.get(name, minv)}
                return data

        @self.app.route('/trackbar', methods=['POST'])
        def trackbar_update():
            data = request.get_json()
            name = data['name']
            value = data['value']
            with self._lock:
                self._trackbar_values[name] = value
                if name in self._trackbars:
                    _, _, callback = self._trackbars[name]
                    if callback:
                        callback(value)
            return ('', 204)

    # --- Публичные методы, имитирующие OpenCV highgui ---
    def imshow(self, winname, img):
        """Отправить изображение в окно с именем winname. Если окно не существовало, оно создаётся."""
        with self._lock:
            self._images[winname] = img.copy()

    def createTrackbar(self, trackbarname, winname, value, count, callback=None):
        """Создать трекбар (winname игнорируется, трекбары глобальные)."""
        with self._lock:
            self._trackbars[trackbarname] = (0, count, callback)
            self._trackbar_values[trackbarname] = value

    def getTrackbarPos(self, trackbarname):
        """Получить текущее значение трекбара."""
        with self._lock:
            return self._trackbar_values.get(trackbarname, 0)

    def waitKey(self, delay=1):
        """Задержка в миллисекундах (имитация, клавиши не обрабатываются)."""
        time.sleep(delay / 1000.0)
        return -1

    def destroyAllWindows(self):
        """Остановить сервер (опционально)."""
        self._running = False
        # Примечание: Flask в daemon-потоке завершится при выходе из основного потока.