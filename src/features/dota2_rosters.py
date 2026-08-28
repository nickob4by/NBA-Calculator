import pandas as pd

DOTA2_ROSTERS = {
    1001: { # NAVI
        'pos1': {'name': 'Yuragi', 'kda': 4.1, 'gpm': 680, 'rating': 86, 'heroes': 'Morphling, Terrorblade, Sven'},
        'pos2': {'name': 'Sanctity-', 'kda': 3.9, 'gpm': 620, 'rating': 85, 'heroes': 'Storm Spirit, Puck, Ember Spirit'},
        'pos3': {'name': 'nefrit', 'kda': 3.4, 'gpm': 510, 'rating': 83, 'heroes': 'Centaur, Mars, Dragon Knight'},
        'pos4': {'name': 'Zayac', 'kda': 3.1, 'gpm': 410, 'rating': 87, 'heroes': 'Techies, Nyx, Mirana'},
        'pos5': {'name': 'Malady', 'kda': 2.8, 'gpm': 340, 'rating': 84, 'heroes': 'Disruptor, Clockwerk, Chen'}
    },
    1002: { # NAVI Junior (EPL Masters)
        'pos1': {'name': 'gotthejuice', 'kda': 4.0, 'gpm': 670, 'rating': 84, 'heroes': 'Medusa, Luna, Gyrocopter'},
        'pos2': {'name': 'Niku', 'kda': 4.2, 'gpm': 640, 'rating': 86, 'heroes': 'Invoker, Pangolier, Void Spirit'},
        'pos3': {'name': 'pma', 'kda': 3.3, 'gpm': 500, 'rating': 82, 'heroes': 'Beastmaster, Brewmaster, Enigma'},
        'pos4': {'name': 'daze', 'kda': 3.0, 'gpm': 390, 'rating': 83, 'heroes': 'Earthshaker, Tiny, Tusk'},
        'pos5': {'name': 'Riddys', 'kda': 2.7, 'gpm': 330, 'rating': 81, 'heroes': 'Shadow Demon, Bane, Treant'}
    },
    1003: { # MOUZ (EPL Masters)
        'pos1': {'name': 'Ulnit', 'kda': 4.3, 'gpm': 690, 'rating': 87, 'heroes': 'Phantom Assassin, Weaver, Ursa'},
        'pos2': {'name': 'MidOne', 'kda': 3.8, 'gpm': 610, 'rating': 85, 'heroes': 'Kunkka, Ember Spirit, Tiny'},
        'pos3': {'name': 'Force', 'kda': 3.5, 'gpm': 520, 'rating': 84, 'heroes': 'Mars, Tidehunter, Dark Seer'},
        'pos4': {'name': 'NARMAN', 'kda': 3.2, 'gpm': 400, 'rating': 83, 'heroes': 'Hoodwink, Rubick, Marci'},
        'pos5': {'name': 'Immersion', 'kda': 2.9, 'gpm': 350, 'rating': 85, 'heroes': 'Undying, Oracle, Shadow Shaman'}
    },
    1004: { # 1win (EPL Masters)
        'pos1': {'name': 'Munkushi~', 'kda': 4.4, 'gpm': 710, 'rating': 89, 'heroes': 'Luna, Shadow Fiend, Templar'},
        'pos2': {'name': 'CHIRA_JUNIOR', 'kda': 4.1, 'gpm': 630, 'rating': 87, 'heroes': 'Puck, Queen of Pain, Kunkka'},
        'pos3': {'name': 'Cloud', 'kda': 3.6, 'gpm': 530, 'rating': 86, 'heroes': 'Dragon Knight, Sand King, Doom'},
        'pos4': {'name': 'swedenstrong', 'kda': 3.3, 'gpm': 420, 'rating': 88, 'heroes': 'Earth Spirit, Rubick, Batrider'},
        'pos5': {'name': 'RESPECT', 'kda': 2.9, 'gpm': 360, 'rating': 85, 'heroes': 'Clockwerk, Jakiro, Phoenix'}
    },
    1005: { # Alliance
        'pos1': {'name': 'Nande', 'kda': 3.8, 'gpm': 640, 'rating': 82, 'heroes': 'Slark, Juggernaut, Lifestealer'},
        'pos2': {'name': 'MoOz', 'kda': 3.6, 'gpm': 590, 'rating': 81, 'heroes': 'Batrider, Tiny, Earth Spirit'},
        'pos3': {'name': 'dEsKaSo', 'kda': 3.2, 'gpm': 480, 'rating': 80, 'heroes': 'Primal Beast, Timbersaw, Axe'},
        'pos4': {'name': 'dEsire', 'kda': 2.9, 'gpm': 380, 'rating': 81, 'heroes': 'Snapfire, Phoenix, Weaver'},
        'pos5': {'name': 'Lelis', 'kda': 2.8, 'gpm': 330, 'rating': 83, 'heroes': 'Vengeful Spirit, Bane, Chen'}
    },
    1007: { # OG
        'pos1': {'name': 'Timado', 'kda': 4.2, 'gpm': 700, 'rating': 88, 'heroes': 'Morphling, Faceless Void, Monkey King'},
        'pos2': {'name': 'bzm', 'kda': 4.6, 'gpm': 670, 'rating': 92, 'heroes': 'Invoker, Puck, Storm Spirit'},
        'pos3': {'name': 'Wisper', 'kda': 4.1, 'gpm': 580, 'rating': 90, 'heroes': 'Batrider, Slardar, Mars'},
        'pos4': {'name': 'Ari', 'kda': 3.5, 'gpm': 430, 'rating': 88, 'heroes': 'Muerta, Earthshaker, Nyx Assassin'},
        'pos5': {'name': 'Ceb', 'kda': 3.1, 'gpm': 370, 'rating': 91, 'heroes': 'Windranger, Treant, Oracle'}
    },
    1008: { # Tundra Esports (New Post-Shuffle Roster)
        'pos1': {'name': 'Nightfall', 'kda': 4.8, 'gpm': 750, 'rating': 94, 'heroes': 'Morphling, Shadow Fiend, Medusa'},
        'pos2': {'name': 'Lorenof', 'kda': 4.3, 'gpm': 660, 'rating': 90, 'heroes': 'Pangolier, Leshrac, Kunkka'},
        'pos3': {'name': '33', 'kda': 4.4, 'gpm': 600, 'rating': 96, 'heroes': 'Visage, Doom, Beastmaster'},
        'pos4': {'name': 'Saksa', 'kda': 3.7, 'gpm': 450, 'rating': 93, 'heroes': 'Tusk, Rubick, Tiny'},
        'pos5': {'name': 'Whitemon', 'kda': 3.2, 'gpm': 360, 'rating': 91, 'heroes': 'Disruptor, Clockwerk, Shadow Demon'}
    },
    1009: { # Team Falcons
        'pos1': {'name': 'skiter', 'kda': 4.7, 'gpm': 740, 'rating': 94, 'heroes': 'Razor, Chaos Knight, Gyrocopter'},
        'pos2': {'name': 'Malr1ne', 'kda': 4.9, 'gpm': 690, 'rating': 97, 'heroes': 'Timbersaw, Razor, Dragon Knight'},
        'pos3': {'name': 'ATF (Ammar)', 'kda': 4.8, 'gpm': 620, 'rating': 98, 'heroes': 'Mars, Slardar, Huskar, Timber'},
        'pos4': {'name': 'Cr1t-', 'kda': 3.9, 'gpm': 460, 'rating': 96, 'heroes': 'Hoodwink, Dark Willow, Earth Spirit'},
        'pos5': {'name': 'Sneyking', 'kda': 3.3, 'gpm': 380, 'rating': 95, 'heroes': 'Mirana, Shadow Demon, Enchantress'}
    },
    1010: { # Gaimin Gladiators (Watson Transfer)
        'pos1': {'name': 'watson', 'kda': 4.7, 'gpm': 760, 'rating': 94, 'heroes': 'Terrorblade, Morphling, Windranger'},
        'pos2': {'name': 'Quinn', 'kda': 4.6, 'gpm': 680, 'rating': 95, 'heroes': 'Pangolier, Leshrac, Storm Spirit'},
        'pos3': {'name': 'Ace', 'kda': 4.2, 'gpm': 580, 'rating': 94, 'heroes': 'Lone Druid, Brewmaster, Underlord'},
        'pos4': {'name': 'tOfu', 'kda': 3.6, 'gpm': 440, 'rating': 93, 'heroes': 'Rubick, Techies, Mirana'},
        'pos5': {'name': 'Seleri', 'kda': 3.1, 'gpm': 360, 'rating': 93, 'heroes': 'Chen, Enchantress, Ancient Apparition'}
    },
    1011: { # Team Spirit (Satanic & rue New Generation)
        'pos1': {'name': 'Satanic', 'kda': 4.9, 'gpm': 770, 'rating': 95, 'heroes': 'Morphling, Luna, Medusa, Faceless'},
        'pos2': {'name': 'Larl', 'kda': 4.2, 'gpm': 650, 'rating': 92, 'heroes': 'Dragon Knight, Pangolier, Sniper'},
        'pos3': {'name': 'Collapse', 'kda': 4.7, 'gpm': 590, 'rating': 97, 'heroes': 'Magnus, Mars, Spirit Breaker'},
        'pos4': {'name': 'rue', 'kda': 3.5, 'gpm': 430, 'rating': 90, 'heroes': 'Rubick, Tiny, Tusk'},
        'pos5': {'name': 'Miposhka', 'kda': 3.2, 'gpm': 360, 'rating': 95, 'heroes': 'Bane, Enchantress, Disruptor'}
    },
    1012: { # Team Liquid
        'pos1': {'name': 'miCKe', 'kda': 4.8, 'gpm': 740, 'rating': 95, 'heroes': 'Morphling, Bloodseeker, Lina'},
        'pos2': {'name': 'Nisha', 'kda': 5.1, 'gpm': 710, 'rating': 98, 'heroes': 'Puck, Ember Spirit, Invoker, Sand King'},
        'pos3': {'name': 'SabeRLighT-', 'kda': 4.2, 'gpm': 580, 'rating': 91, 'heroes': 'Axe, Beastmaster, Centaur'},
        'pos4': {'name': 'Boxi', 'kda': 3.8, 'gpm': 450, 'rating': 95, 'heroes': 'Tusk, Weaver, Techies'},
        'pos5': {'name': 'Insania', 'kda': 3.3, 'gpm': 370, 'rating': 94, 'heroes': 'Oracle, Rubick, Shadow Demon'}
    },
    1013: { # BetBoom Team
        'pos1': {'name': 'Pure~', 'kda': 4.7, 'gpm': 750, 'rating': 94, 'heroes': 'Doom, Pudge, Faceless Void'},
        'pos2': {'name': 'gpk~', 'kda': 4.6, 'gpm': 680, 'rating': 94, 'heroes': 'Invoker, Storm, Templar Assassin'},
        'pos3': {'name': 'MieRo', 'kda': 4.1, 'gpm': 570, 'rating': 91, 'heroes': 'Enigma, Mars, Centaur'},
        'pos4': {'name': 'Save-', 'kda': 3.8, 'gpm': 450, 'rating': 93, 'heroes': 'Shadow Demon, Hoodwink, Mirana'},
        'pos5': {'name': 'TORONTOTOKYO', 'kda': 3.2, 'gpm': 370, 'rating': 90, 'heroes': 'Clockwerk, Undying, Lich'}
    }
}

def get_team_roster_data(team_id: int) -> dict:
    if team_id in DOTA2_ROSTERS:
        return DOTA2_ROSTERS[team_id]
    # Default generated pro roster
    return {
        'pos1': {'name': 'Core Carry', 'kda': 4.0, 'gpm': 660, 'rating': 84, 'heroes': 'Luna, Morphling, Sven'},
        'pos2': {'name': 'Mid Playmaker', 'kda': 3.9, 'gpm': 620, 'rating': 84, 'heroes': 'Puck, Kunkka, Invoker'},
        'pos3': {'name': 'Offlane Initiator', 'kda': 3.4, 'gpm': 510, 'rating': 83, 'heroes': 'Centaur, Mars, Axe'},
        'pos4': {'name': 'Soft Roamer', 'kda': 3.0, 'gpm': 400, 'rating': 82, 'heroes': 'Rubick, Tiny, Tusk'},
        'pos5': {'name': 'Captain Support', 'kda': 2.7, 'gpm': 340, 'rating': 82, 'heroes': 'Disruptor, Clockwerk, Chen'}
    }

def calculate_roster_composite_rating(team_id: int, standin_penalty: float = 0.0) -> float:
    roster = get_team_roster_data(team_id)
    weights = [0.28, 0.26, 0.22, 0.14, 0.10]
    keys = ['pos1', 'pos2', 'pos3', 'pos4', 'pos5']
    composite = sum(roster[k]['rating'] * w for k, w in zip(keys, weights))
    return round(composite - standin_penalty, 2)
