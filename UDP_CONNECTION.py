import socket
import threading
import queue
import time

def create_socket():
    """Erstellt einen neuen UDP-Socket."""
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_udp_telegram(ip: str, port: int, message: bytes, sock: socket.socket):
    """
    Sendet ein UDP-Telegramm.
    
    :param ip: Ziel-IP-Adresse
    :param port: Ziel-Port
    :param message: Nachricht als Bytes
    :param sock: Offener UDP-Socket
    """
    if sock.fileno() == -1:
        print("Socket wurde geschlossen. Senden abgebrochen.")
        return
    
    try:
        target = (ip, port)
        sock.sendto(message, target)
        #print(f"Gesendet: {message} an {ip}:{port}")
    except OSError as e:
        print(f"Fehler beim Senden: {e}. Neuer Socket wird erstellt.")
        sock.close()

def receive_udp_response(sock: socket.socket, stop_event: threading.Event, response_queue: queue.Queue):
    """
    Wartet asynchron auf eine UDP-Antwort und prüft auf Timeout.
    
    :param sock: Offener UDP-Socket
    :param stop_event: Event zum Beenden des Empfangs-Threads
    :param response_queue: Queue zum Speichern empfangener Antworten
    """
    last_received_time = time.time()  # Zeitpunkt des letzten Empfangs
    print(f"Zeit letzte Antwort {last_received_time}")

    while not stop_event.is_set():
        try:
            sock.settimeout(1)  # Setze Timeout für `recvfrom()`
            response, addr = sock.recvfrom(1024)  # Antwort empfangen
            print(f"Antwort von {addr}: {response}")
            response_queue.put(response)
            last_received_time = time.time()  # Aktualisiere den Empfangszeitpunkt

        except socket.timeout:
            # Prüfen, ob 5 Sekunden ohne Antwort vergangen sind
            if time.time() - last_received_time >= 5:
                print("Fehler: Kein Telegramm innerhalb von 5 Sekunden empfangen!")
                last_received_time = time.time()  # Zeit zurücksetzen, um mehrfachen Fehler zu vermeiden

        except OSError as e:
            print(f"Socket-Fehler beim Empfangen: {e}.")
            break  # Fehler aufgetreten, beende Thread

    print("Empfangs-Thread beendet.")  # Wird ausgegeben, wenn `stop_event` gesetzt wird