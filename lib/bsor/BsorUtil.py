
from .Decoder import *
from .Bsor import *
import json
import requests
import io
import logging

logger = logging.getLogger(__name__)
def download_Bsor(id: int) -> Bsor:
    # Download the file from the server
    with requests.get(f'https://api.beatleader.xyz/score/{id}') as r:
        r.raise_for_status()
        parsed = json.loads(r.content)
        replay_location = parsed['replay']
    with requests.get(replay_location) as replay:
        replay.raise_for_status()
        return make_bsor(io.BufferedReader(io.BytesIO(replay.content)))

def song_from_data(d):
    return {
        'songName': d['song']['name'],
        'levelAuthorName': d['song']['author'],
        'hash': d['song']['hash'],
        'levelid': 'custom_level_' + d['song']['hash'],
        'difficulties': [
            {
                'characteristic': d['difficulty']['modeName'],
                'name': d['difficulty']['difficultyName'],
            }
        ]
    }


def check_song(song_data, playlist, count, stars_from, stars_to, unplayed_player=None):
    if len(playlist['songs']) >= count:
        return False
    if stars_from and song_data['difficulty']['stars'] < stars_from:
        return False
    if stars_to and song_data['difficulty']['stars'] > stars_to:
        return False
    if unplayed_player:
        p_s = f'https://api.beatleader.xyz/player/{unplayed_player}/scorevalue/{song_data["song"]["hash"]}/{song_data["difficulty"]["difficultyName"]}/{song_data["difficulty"]["modeName"]}'
        logger.info(p_s)
        with requests.get(p_s) as pr:
            if pr.status_code == 200:
                if int(pr.content) > 0:
                    return False
    return True

def img_url_to_b64(url: str) -> str:
    with requests.get(url) as img:
        img.raise_for_status()
        import base64
        content = io.BytesIO(img.content)
        return base64.b64encode(content.read()).decode('utf-8')

def createStatsDB(player: int, filename: str):
    import sqlite3
    import time
    conn = sqlite3.connect(filename)
    #get unix timestamp of now
    now = int(time.time())
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scores
                    (id text,
                        id_user text, 
                        date text, 
                        
                        id_leaderboard TEXT,
                        id_song_hash TEXT, 
                         
                        difficulty_name TEXT, 
                        mode_name TEXT, 
                        stars REAL, 
                        passRating REAL, 
                        accRating REAL,
                        techRating REAL,
                        duration INTEGER,
                        
                        accLeft REAL,
                        accRight REAL,
                        pp REAL,
                        passPP REAL,
                        accPP REAL,
                        techPP REAL,
                        fcAcc REAL,
                        fcPP REAL,
                        modifiers TEXT,
                        badCuts INTEGER,
                        missedNotes INTEGER,
                        bombCuts INTEGER,
                        wallHits INTEGER,
                        maxCombo INTEGER,
                        weight REAL,
                        weight_reference_date text,
                        replay text
                    )''')
    page = 0
    count = 50
    maxP = 5000
    while page <= maxP//count+1:
        page += 1
        print(page)
        s = f'https://api.beatleader.xyz/player/{player}/scores?page={page}&count={count}'
        with requests.get(s) as r:
            r.raise_for_status()
            parsed = json.loads(r.content)
            maxP = int(parsed['metadata']['total'])

            for d in parsed['data']:
                id = d['id']
                playerId = d['playerId']
                date = d['timeset']
                id_leaderboard = d['leaderboard']['id']
                id_song = d['leaderboard']['song']['hash']
                difficulty_name = d['leaderboard']['difficulty']['difficultyName']
                mode_name = d['leaderboard']['difficulty']['modeName']
                stars = d['leaderboard']['difficulty']['stars']
                passRating = d['leaderboard']['difficulty']['passRating']
                accRating = d['leaderboard']['difficulty']['accRating']
                techRating = d['leaderboard']['difficulty']['techRating']
                duration = d['leaderboard']['difficulty']['duration']
                accLeft = d['accLeft']
                accRight = d['accRight']
                pp = d['pp']
                passPP = d['passPP']
                accPP = d['accPP']
                techPP = d['techPP']
                fcAcc = d['fcAccuracy']
                fcPP = d['fcPp']
                modifiers = d['modifiers']

                badCuts = d['badCuts']
                missedNotes = d['missedNotes']
                bombCuts = d['bombCuts']
                wallsHit = d['wallsHit']
                maxCombo = d['maxCombo']
                weight = d['weight']
                weight_reference_date = now
                replay = d['replay']

                #query if id already exists
                c.execute(f"SELECT * FROM scores WHERE id='{id}'")
                if c.fetchone() is None:
                    c.execute(f"INSERT INTO scores VALUES ("
                              f"'{id}', '{playerId}', '{date}', "
                              f"'{id_leaderboard}', '{id_song}', '{difficulty_name}', '{mode_name}', {stars}, {passRating}, {accRating}, {techRating}, {duration}, "
                              f"{accLeft}, {accRight}, {pp}, {passPP}, {accPP}, {techPP}, {fcAcc}, {fcPP}, '{modifiers}', {badCuts}, {missedNotes}, {bombCuts}, {wallsHit}, {maxCombo}, {weight}, '{weight_reference_date}', '{replay}')")
        conn.commit()

    conn.close()

def unplayed_list(unplayed_player: int, count: int = 20, stars_from: int = None, stars_to: int = None) -> dict:

    s = f'https://api.beatleader.xyz/player/{unplayed_player}'
    with requests.get(s) as r:
        r.raise_for_status()
        parsed = json.loads(r.content)
        unplayed_player_name = parsed['name']

    playlist = {
        'playlistTitle': f'unplayed songs for {unplayed_player_name}',
        'playlistAuthor': 'Schippi',
        'songs': [],
        'image': img_url_to_b64(parsed['avatar'])
    }

    # Get the playlist from the server
    #
    page = 1
    while len(playlist['songs']) < count:
        s = f'https://api.beatleader.xyz/leaderboards?leaderboardContext=general&page={page}&count={min(count,50)}' \
            f'&type=ranked&sortBy=stars&order=asc&allTypes=0&allRequirements=0'
        if stars_from:
            s += f'&stars_from={stars_from}'
        if stars_to:
            s += f'&stars_to={stars_to}'

        logger.info(s)
        with requests.get(s) as r:
            r.raise_for_status()
            # Parse the JSON response
            parsed = json.loads(r.content)
            if len(parsed['data']) == 0:
                logger.warning(f'no more songs found after page {page}')
                break
            for d in parsed['data']:
                if check_song(d, playlist, count, stars_from, stars_to, unplayed_player):
                    playlist['songs'].append(song_from_data(d))
        logger.info(f'songs in playlist after page {page}: {len(playlist["songs"])}')
        page += 1

    return playlist
    # and return the playlist

"""
    Get the playlist for a clan
    :param clan: the clan tag
    :param count: the number of songs to get, standard is 20
    :param imageb64: the base64 encoded image for the playlist, if None, it the clan-icon will be downloaded
    :param unplayed_player: the selected songs should not have been played by this player, optional 
    :param stars_from: the minimum stars for the songs, optional
    :param stars_to: the maximum stars for the songs, optional
    :param include_to_hold: include the to hold maps, if False only toConquer maps will be included
"""
def clan_playlist(clan: str, count: int = 20, imageb64: str = None, unplayed_player: int = None,
                  stars_from: int = None, stars_to: int = None, include_to_hold: bool = False,
                  author: str = 'Schippi') -> dict:
    clan = clan[:5]
    playlist = {
        'playlistTitle': f'contested maps for {clan}',
        'playlistAuthor': author,
        'songs': [],
        'image': imageb64
    }

    def get_image(parsed_response):
        if imageb64:
            return imageb64
        img_url = parsed_response['data'][0]['clan']['icon']
        try:
            return img_url_to_b64(img_url)
        except:
            return ''

    # Get the playlist from the server
    divisor = 2 if include_to_hold else 1
    page = 1
    while len(playlist['songs']) < count:
        s = f'https://api.beatleader.xyz/clan/{clan}/maps?page={page}&count={count//divisor}&sortBy=toconquer&order=0'
        logger.info(s)
        with requests.get(s) as r:
            r.raise_for_status()
            # Parse the JSON response
            parsed = json.loads(r.content)
            playlist['image'] = get_image(parsed)
            if len(parsed['data']) == 0:
                logger.warning(f'no more songs found after page {page}')
                break
            for d in parsed['data']:
                if check_song(d['leaderboard'], playlist, count, stars_from, stars_to, unplayed_player):
                    playlist['songs'].append(song_from_data(d['leaderboard']))

        if include_to_hold:
            s = f'https://api.beatleader.xyz/clan/{clan}/maps?page={page}&count={count//divisor}&sortBy=tohold&order=0'
            logging.info(s)
            with requests.get(s) as r:
                r.raise_for_status()
                # Parse the JSON response
                parsed = json.loads(r.content)
                for d in parsed['data']:
                    if check_song(d['leaderboard'], playlist, count, stars_from, stars_to, unplayed_player):
                        playlist['songs'].append(song_from_data(d))
        logger.info(f'songs in playlist after page {page}: {len(playlist["songs"])}')
        page += 1

    return playlist
    # and return the playlist
