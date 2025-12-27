"""Domain-specific rules and mappings for the rental marketplace."""

# Mapping from URL slug to main_category name
SLUG_TO_CAT = {
    'kolyaski': 'Коляски',
    'avtokresla-avtolyulki': 'Автокресла, автолюльки',
    'krovatki-manezhi': 'Кроватки, манежи',
    'kacheli-shezlongi': 'Электрокачели',
    'vesy': 'Весы',
    'stulchiki-dlya-kormleniya': 'Стульчики для кормления',
    'ergoryukzaki': 'Эргорюкзаки',
    'kolyaski-dlya-puteshestviy': 'Коляски для путешествий',
    'razvivayuschie-igrushki': 'Развивающие игрушки',
    'velosipedy': 'Велосипеды',
    'videonyani': 'Видеоняни',
    'begovely': 'Беговелы',
    'progulochnye-kolyaski': 'Прогулочные коляски',
    'kolyaski-yoyo': 'Коляски YoYo',
    'samokaty': 'Самокаты',
    'kokony-dlya-novorozhdennyh': 'Коконы для новорожденных',
    'avtokresla-9-36-kg': 'Автокресла 9-36 кг',
}

# Biological Progression (Deterministic Transitions)
# If user views K, also suggest V
BIOLOGICAL_TRANSITIONS = {
    # YoYo 0+ (Newborn) -> YoYo 6+ (Infant)
    463480240: [463480227, 463480255, 463480468, 463480466, 463480223], # BabyZen YoYo 6+ variants
    463480226: [463480227, 463480255, 463480468, 463480466, 463480223],
    
    # Doona+ Car Seat (0-13kg) -> Toddler Car Seat (9-36kg or 15-36kg)
    463480493: [463480322, 463480242, 463480693, 463480694, 463480221], 
}

# Brand Synergy Groups
BRAND_GROUPS = {
    'BabyZen': [463480240, 463480227, 463480237, 463480234, 463480251, 463480226, 463480223, 463480468, 463480466],
    'Bugaboo': [1228501633, 463480252, 463480254],
    'Doona': [463480493],
    'Britax Romer': [463480322, 463480242, 463480221, 463480230, 463480243],
}

def get_biological_boost(product_ids):
    """Return additional product IDs based on biological progression rules."""
    boosted = []
    for pid in product_ids:
        if pid in BIOLOGICAL_TRANSITIONS:
            boosted.extend(BIOLOGICAL_TRANSITIONS[pid])
    return boosted

def get_category_from_slug(slug):
    """Return category name from URL slug."""
    return SLUG_TO_CAT.get(slug, None)
