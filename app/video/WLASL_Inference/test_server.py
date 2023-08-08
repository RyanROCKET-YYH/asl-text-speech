# server.py

import asyncio
import websockets

async def handle_client(websocket, path):
    try:
        async for data in websocket:
            await websocket.send(data)  # Echo the received data back to the client
    except websockets.ConnectionClosedError:
        print(f"Connection closed by {websocket.remote_address}")

async def main():
    # Start the WebSocket server
    async with websockets.serve(handle_client, '0.0.0.0', 4000):
        print('WebSocket server started.')
        await asyncio.Future()  # Run forever

# Run the main event loop
if __name__ == '__main__':
    asyncio.run(main())

