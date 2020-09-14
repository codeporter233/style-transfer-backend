from os import path

from flask import Flask, redirect, request

from pywsgi import WSGIServer
from startup import load_plugin

app = Flask(__name__)


@app.errorhandler(404)
def page_not_found(e):
    app.logger.warning("由于 404 重定向 %s", request.url)
    return redirect('https://www.baidu.com')


# 循环引入api
def initializer():
    plugins = load_plugin.load_plugins(
        path.join(path.dirname(__file__), "plugins"),
        "plugins"
    )
    for i in plugins:
        app.register_blueprint(i.api)


if __name__ == '__main__':
    initializer()
    http_server = WSGIServer(('0.0.0.0', 10001), app)
    http_server.serve_forever()
