def read_data(file_path):
    dimension = 0
    capacity = 0
    nodes = {}  
    depots = [] 

    current_section = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line or line == 'EOF':
                continue

            # Xác định section đang đọc
            if line.startswith('NODE_COORD_SECTION'):
                current_section = 'NODE_COORD_SECTION'
                continue
            elif line.startswith('DEMAND_SECTION'):
                current_section = 'DEMAND_SECTION'
                continue
            elif line.startswith('DEPOT_SECTION'):
                current_section = 'DEPOT_SECTION'
                continue

            # Xử lý trích xuất dữ liệu
            if current_section is None:
                if ':' in line:
                    key, value = [x.strip() for x in line.split(':', 1)]
                    if key == 'DIMENSION':
                        dimension = int(value)
                    elif key == 'CAPACITY':
                        capacity = int(value)
                        
            elif current_section == 'NODE_COORD_SECTION':
                parts = line.split()
                node_id = int(parts[0])
                if node_id not in nodes:
                    nodes[node_id] = {}
                nodes[node_id]['x'] = float(parts[1])
                nodes[node_id]['y'] = float(parts[2])
                
            elif current_section == 'DEMAND_SECTION':
                parts = line.split()
                node_id = int(parts[0])
                if node_id not in nodes:
                    nodes[node_id] = {}
                nodes[node_id]['demand'] = int(parts[1])
                
            elif current_section == 'DEPOT_SECTION':
                parts = line.split()
                depot_id = int(parts[0])
                if depot_id != -1:
                    depots.append(depot_id)

    return dimension, capacity, nodes