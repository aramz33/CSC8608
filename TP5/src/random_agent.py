import gymnasium as gym
from PIL import Image

env = gym.make("LunarLander-v3", render_mode="rgb_array")

print("Espace d'observation (Capteurs) :", env.observation_space)
print("Espace d'action (Moteurs) :", env.action_space)

obs, info = env.reset()
done = False
frames = []

total_reward = 0.0
main_engine_uses = 0
side_engine_uses = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    if action == 2:
        main_engine_uses += 1
    elif action in [1, 3]:
        side_engine_uses += 1

    frame = env.render()
    frames.append(Image.fromarray(frame))

    done = terminated or truncated

env.close()

if reward == -100:
    issue = "CRASH DÉTECTÉ 💥"
elif reward == 100:
    issue = "ATTERRISSAGE RÉUSSI 🏆"
else:
    issue = "TEMPS ÉCOULÉ OU SORTIE DE ZONE ⚠️"

print("\n--- RAPPORT DE VOL ---")
print(f"Issue du vol : {issue}")
print(f"Récompense totale cumulée : {total_reward:.2f} points")
print(f"Allumages moteur principal : {main_engine_uses}")
print(f"Allumages moteurs latéraux : {side_engine_uses}")
print(f"Durée du vol : {len(frames)} frames")

if frames:
    frames[0].save('random_agent.gif', save_all=True, append_images=frames[1:], duration=30, loop=0)
    print("Vidéo de la télémétrie sauvegardée sous 'random_agent.gif'")
