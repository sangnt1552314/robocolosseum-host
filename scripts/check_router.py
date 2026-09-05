import socket
import ssl
import time

HOST = "191.222.219.43"
PORT = 8443
TIMEOUT = 5

print(f"Testing {HOST}:{PORT} ...")

start = time.time()

try:
    sock = socket.create_connection((HOST, PORT), timeout=TIMEOUT)
    print(f"TCP connection: OK ({time.time() - start:.2f}s)")

    context = ssl.create_default_context()

    try:
        tls_sock = context.wrap_socket(sock, server_hostname=HOST)
        print("TLS handshake: OK")
        print("TLS version:", tls_sock.version())
        print("Cipher:", tls_sock.cipher())
        tls_sock.close()

    except ssl.SSLError as e:
        print("TCP works, but TLS handshake failed:")
        print(e)

except socket.timeout:
    print("TCP connection: TIMEOUT")

except ConnectionRefusedError:
    print("TCP connection: REFUSED")

except OSError as e:
    print("TCP connection failed:")
    print(e)