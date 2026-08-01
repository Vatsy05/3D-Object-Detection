"""
Class registry v2 — 60-class taxonomy (10 furniture + 50 military).
Phase 9. Import this from the scene generator instead of the inline Phase 8 dicts.

Sizes = target longest dimension in meters (generator scales longest extent to this).
Furniture + original 17 military values unchanged from Phase 8.
"""

FURNITURE_CLASSES = [
    'bathtub', 'bed', 'bookshelf', 'chair', 'desk',
    'dresser', 'night_stand', 'sofa', 'table', 'toilet',
]

MILITARY_CLASSES = [
    # original 17 (Phase 8)
    'ammo_box', 'binoculars', 'combat_knife', 'flashlight', 'gas_mask',
    'hand_grenade', 'helmet', 'magazine', 'military_radio', 'pistol',
    'rifle', 'rocket_launcher', 'shotgun', 'sniper_rifle',
    'tactical_backpack', 'tactical_vest', 'wire_cutter',
    # new 33 (Phase 9)
    'axe', 'barbed_wire_coil', 'baton', 'canteen', 'claymore_mine',
    'concrete_barrier', 'crossbow', 'duffel_bag', 'entrenching_shovel',
    'field_telephone', 'first_aid_kit', 'flare_gun', 'fuel_drum',
    'grenade_launcher', 'hedgehog', 'jerry_can', 'machete', 'machine_gun',
    'military_boots', 'military_cot', 'military_drone', 'military_shield',
    'mortar_tube', 'night_vision_goggles', 'propane_tank', 'rifle_case',
    'sandbag', 'smoke_grenade', 'stretcher', 'submachine_gun',
    'tank_mine', 'tank_shell', 'weapon_rack',
]

CLASS_NAMES = FURNITURE_CLASSES + MILITARY_CLASSES
NUM_CLASSES = len(CLASS_NAMES)  # 60
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

CLASS_REAL_SIZE = {
    # furniture (unchanged, Phase 8)
    'bed': 2.0, 'table': 1.5, 'sofa': 2.0, 'chair': 0.55, 'toilet': 0.6,
    'desk': 1.4, 'dresser': 1.0, 'night_stand': 0.55, 'bookshelf': 0.85, 'bathtub': 1.6,
    # military, original 17 (unchanged, Phase 8)
    'ammo_box': 0.35, 'binoculars': 0.22, 'combat_knife': 0.30, 'flashlight': 0.18,
    'gas_mask': 0.28, 'hand_grenade': 0.12, 'helmet': 0.28, 'magazine': 0.18,
    'military_radio': 0.30, 'pistol': 0.22, 'rifle': 0.95, 'rocket_launcher': 1.20,
    'shotgun': 0.95, 'sniper_rifle': 1.20, 'tactical_backpack': 0.55,
    'tactical_vest': 0.50, 'wire_cutter': 0.25,
    # military, new 33 (Phase 9) — REVIEW THESE, they are proposed defaults
    'axe': 0.60,                 # tactical/felling axe
    'barbed_wire_coil': 0.90,    # concertina coil diameter
    'baton': 0.55,               # expandable baton, extended
    'canteen': 0.20,
    'claymore_mine': 0.22,       # M18A1 width
    'concrete_barrier': 2.00,    # Jersey barrier section
    'crossbow': 0.75,
    'duffel_bag': 0.80,
    'entrenching_shovel': 0.60,
    'field_telephone': 0.30,
    'first_aid_kit': 0.30,
    'flare_gun': 0.25,
    'fuel_drum': 0.90,           # 55-gal drum height
    'grenade_launcher': 0.75,    # M79 / standalone GL
    'hedgehog': 1.40,            # Czech hedgehog
    'jerry_can': 0.47,
    'machete': 0.65,
    'machine_gun': 1.25,         # GPMG w/ bipod
    'military_boots': 0.32,      # single boot length
    'military_cot': 1.90,
    'military_drone': 0.90,      # quadcopter diagonal
    'military_shield': 1.30,     # ballistic shield height
    'mortar_tube': 1.30,         # 81mm w/ bipod
    'night_vision_goggles': 0.20,
    'propane_tank': 0.60,
    'rifle_case': 1.20,
    'sandbag': 0.65,
    'smoke_grenade': 0.15,
    'stretcher': 2.10,
    'submachine_gun': 0.60,
    'tank_mine': 0.33,           # AT mine diameter
    'tank_shell': 0.90,          # 120mm round
    'weapon_rack': 1.80,
}

# Longest axis horizontal (lying flat on floor/table)
FLAT_CLASSES = {
    # Phase 8
    'bed', 'sofa', 'table', 'desk', 'bathtub',
    'pistol', 'rifle', 'shotgun', 'sniper_rifle', 'combat_knife',
    'rocket_launcher', 'magazine', 'hand_grenade', 'wire_cutter',
    'ammo_box', 'binoculars',
    # Phase 9
    'machine_gun', 'submachine_gun', 'grenade_launcher', 'flare_gun',
    'crossbow', 'machete', 'axe', 'baton', 'entrenching_shovel',
    'rifle_case', 'stretcher', 'military_cot', 'sandbag',
    'concrete_barrier', 'tank_shell', 'tank_mine', 'claymore_mine',
    'duffel_bag', 'first_aid_kit', 'military_drone',
}

# Longest axis vertical (standing upright)
TALL_CLASSES = {
    # Phase 8
    'chair', 'bookshelf', 'dresser', 'night_stand', 'toilet',
    'helmet', 'gas_mask', 'flashlight', 'military_radio',
    'tactical_backpack', 'tactical_vest',
    # Phase 9
    'fuel_drum', 'propane_tank', 'jerry_can', 'canteen', 'weapon_rack',
    'military_shield', 'mortar_tube', 'field_telephone', 'military_boots',
    'smoke_grenade',
}
# neither set: barbed_wire_coil, hedgehog, night_vision_goggles (roughly isotropic)

# Classes with longest dim < 0.35m — targets for small-object handling
SMALL_CLASSES = {c for c, s in CLASS_REAL_SIZE.items() if s < 0.35}

# Meshes flagged in the Phase 9 load-check for MANUAL CONTENT REVIEW:
#   rifle/silver_strike_estoc.glb          <- an estoc is a SWORD, wrong class: remove or move
#   pistol/sci-fi_pistol.glb               <- sci-fi styling, hurts realism
#   sniper_rifle/sniper_rifle_futuristic.glb <- futuristic styling
#   baton/nightwing_batons.glb             <- fictional dual batons (one mesh, two objects?)
#   tank_shell/pixel_tank_shell_pbr.glb    <- pixel-art styling

if __name__ == '__main__':
    assert NUM_CLASSES == 60, NUM_CLASSES
    missing = [c for c in CLASS_NAMES if c not in CLASS_REAL_SIZE]
    assert not missing, f'missing sizes: {missing}'
    overlap = FLAT_CLASSES & TALL_CLASSES
    assert not overlap, f'flat/tall overlap: {overlap}'
    unknown = (FLAT_CLASSES | TALL_CLASSES) - set(CLASS_NAMES)
    assert not unknown, f'unknown class in flat/tall: {unknown}'
    print(f'{NUM_CLASSES} classes OK; {len(SMALL_CLASSES)} small (<35cm):')
    print(sorted(SMALL_CLASSES))
