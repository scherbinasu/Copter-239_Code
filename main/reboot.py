import asyncio
from robot import robots


drone = robots.Drone()


async def main():
    async with drone:
        await drone.drone.action.reboot()


if __name__ == '__main__':
    asyncio.run(main())
