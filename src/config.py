from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 9=Bundesliga, 11=Liga, 7=Ligue1, 2=PL, 12=SerieA
    COMPETITION_IDS = [9, 11, 7, 2, 12] 
    SEASON_ID = 27 # 2015/2016

    CATEGORICAL_COLS = [
        'type_name', 
        'sub_type_name', 
        'outcome_name', 
        'body_part_name', 
        'technique_name', 
        'play_pattern_name'
    ]
    
    BOOLEAN_COLS = [
        'under_pressure', 
        'counterpress', 
        'is_possession_team' # Calcolata custom
    ]
    
    NUMERICAL_COLS = [
        'duration',
        'x', 'y', 
        'end_x', 'end_y'
    ]
    
    META_COLS = [

        'match_id', 
        'id', 
        'index', 
        'period', 
        'timestamp', 
        'possession', 
        'player_id', 
        'team_id'
    ]

    IGNORED_EVENTS = {
        '50/50',
        'Half Start',
        'Half End',
        'Referee Ball-Drop',
        'Player On',
        'Player Off',
        'Tactical Shift',
        'Injury Stoppage',
        'Substitution',
        'Starting XI',
        'Ball Receipt',
    }
    MIN_CARRY_DISTANCE = 3.0
    GRID_WIDTH = 120
    GRID_HEIGHT = 80
    MAX_SEQ_LEN = 128
    HIDDEN_SIZE = 256
    LAYERS = 6
    HEADS = 8
    MAX_LEN = 128
