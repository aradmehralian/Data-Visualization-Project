import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    import json

    return KMeans, json, mo, np, pd


@app.cell
def _(pd):
    df_env = pd.read_csv("data/oumalik_environmental_data.csv", parse_dates=["date"])
    return (df_env,)


@app.cell
def _(df_env):
    df_env.head()
    return


@app.cell
def _(df_env):
    df_env["cover_total"].min()
    return


@app.cell
def _(df_env, np):
    df_env["cover_total"] = df_env["cover_total"].replace(-9999, np.nan)
    return


@app.cell
def _(df_env):
    median_cover = df_env["cover_total"].median()
    return (median_cover,)


@app.cell
def _(df_env, median_cover):
    df_env["cover_total"] = df_env["cover_total"].fillna(median_cover)
    return


@app.cell
def _(pd):
    df_species_raw = pd.read_csv("data/oumalik_species_data.csv", encoding="latin1", header=None)
    return (df_species_raw,)


@app.cell
def _(df_species_raw):
    df_species_raw.head(n=10)
    return


@app.cell
def _(df_species_raw):
    species_names = df_species_raw.iloc[3:, 0].values
    return (species_names,)


@app.cell
def _(species_names):
    species_names
    return


@app.cell
def _(df_species_raw):
    plot_ids = df_species_raw.iloc[1, 3:].values.astype(float).astype(int)
    return (plot_ids,)


@app.cell
def _(plot_ids):
    plot_ids
    return


@app.cell
def _(df_species_raw, np):
    species_matrix = np.nan_to_num(df_species_raw.iloc[3:, 3:].values.astype(float))
    return (species_matrix,)


@app.cell
def _(species_matrix):
    species_matrix
    return


@app.cell
def _(pd, plot_ids, species_matrix):
    df_species = pd.DataFrame(species_matrix, columns=plot_ids.astype(str))
    return (df_species,)


@app.cell
def _(df_species):
    df_species
    return


@app.cell
def _(df_species, species_names):
    df_species.insert(0, "species_name", species_names)
    return


@app.cell
def _(df_species):
    df_species
    return


@app.cell
def _(df_species):
    df_species.iloc[:, 1:].apply(lambda x: sum(x != 0)).max()
    return


@app.cell
def _(df_species):
    df_species_numeric = df_species.set_index("species_name")
    df_species_numeric
    return (df_species_numeric,)


@app.cell
def _(df_species_numeric):
    # converting the oridinal values to binary (presence or absence of species)
    presence_absence = df_species_numeric > 0
    return (presence_absence,)


@app.cell
def _(presence_absence):
    unique_plants_per_plot = presence_absence.sum()
    return (unique_plants_per_plot,)


@app.cell
def _(unique_plants_per_plot):
    unique_plants_per_plot
    return


@app.cell
def _(unique_plants_per_plot):
    df_diversity = unique_plants_per_plot.reset_index()
    return (df_diversity,)


@app.cell
def _(df_diversity):
    df_diversity
    return


@app.cell
def _(df_diversity):
    df_diversity.columns = ["plot_number", "unique_plants"]
    df_diversity
    return


@app.cell
def _(df_env):
    df_env["plot_number"] = df_env["plot_number"].astype(str)
    return


@app.cell
def _(df_diversity, df_env, pd):
    df_final = pd.merge(df_diversity, df_env, on="plot_number", how="left")
    return (df_final,)


@app.cell
def _(df_final):
    df_final
    return


@app.cell
def _(df_final):
    df_final.columns = df_final.columns.str.strip()
    return


@app.cell
def _(df_final, np):
    df_final["disturbance_score"] = np.clip(df_final["disturbance_score"], min=1, max=10)
    return


@app.cell
def _(df_diversity):
    MAX_DIVERSITY = df_diversity["unique_plants"].max()
    print(MAX_DIVERSITY)
    return (MAX_DIVERSITY,)


@app.cell
def _(df_final):
    df_final["cover_total"] = df_final["cover_total"] / 100
    return


@app.cell
def _(MAX_DIVERSITY, df_final):
    df_final["diversity_ratio"] = df_final["unique_plants"] / MAX_DIVERSITY
    return


@app.cell
def _(df_final):
    df_final["concern_score"] = 2 - (df_final["diversity_ratio"] + df_final["cover_total"])
    return


@app.cell
def _(df_final):
    df_final[["plot_number", "unique_plants", "cover_total", "concern_score"]].sort_values(by="concern_score", ascending=False)
    return


@app.cell
def _(df_final):
    df_final["cover_total"].min()
    return


@app.cell
def _(df_final):
    df_final["concern_score"].min()
    return


@app.cell
def _(df_final):
    X = df_final["concern_score"].values.reshape(-1, 1)
    return (X,)


@app.cell
def _(KMeans):
    kmeans = KMeans(n_clusters=3, random_state=7)
    return (kmeans,)


@app.cell
def _(X, df_final, kmeans):
    df_final["raw_cluster"] = kmeans.fit_predict(X)
    return


@app.cell
def _(df_final):
    df_final["raw_cluster"]
    return


@app.cell
def _(kmeans):
    centers = kmeans.cluster_centers_.flatten()
    return (centers,)


@app.cell
def _(centers):
    centers
    return


@app.cell
def _(centers, np):
    sorted_indices = np.argsort(centers)
    sorted_indices
    return (sorted_indices,)


@app.cell
def _(sorted_indices):
    color_mapping = {
        sorted_indices[0]: 'green',
        sorted_indices[1]: 'yellow',
        sorted_indices[2]: 'red'
    }
    return (color_mapping,)


@app.cell
def _(color_mapping, df_final):
    df_final["concern_tier"] = df_final["raw_cluster"].map(color_mapping)
    return


@app.cell
def _(df_final):
    df_final.groupby('concern_tier')['concern_score'].agg(['min', 'max', 'count'])
    return


@app.cell
def _(df_final):
    df_final["disturbance_type"]
    return


@app.cell
def _(df_final, pd):
    animal_cols = [
        'disturbance_caribou', 'disturbance_microtine', 'disturbance_squirrel', 
        'disturbance_ptarmigan', 'disturbance_birds', 'disturbance_insects'
    ]

    label_map = {
        'disturbance_caribou': 'Caribou', 'disturbance_microtine': 'Rodents',
        'disturbance_squirrel': 'Squirrels', 'disturbance_ptarmigan': 'Ptarmigan',
        'disturbance_birds': 'Other Birds', 'disturbance_insects': 'Insects'
    }

    map_data = []

    for index, row in df_final.iterrows():
        lat, lon = row.get('latitude'), row.get('longitude')
        if pd.isna(lat) or pd.isna(lon) or lat == -9999 or lon == -9999:
            continue 
        
        # Build the animal scores array using the clean columns
        animal_data = []
        for col in animal_cols:
            val = row.get(col, 0)
            if pd.notna(val) and val > 0 and val != -9999:
                animal_data.append({
                    "animal": label_map[col],
                    "score": int(val)
                })
        animal_data = sorted(animal_data, key=lambda x: x['score'], reverse=True)

        map_data.append({
            "plot_id": str(row.get('plot_number', index)),
            "latitude": float(lat),
            "longitude": float(lon),
            "releve_area": float(row.get('releve_area', 25)),
            "releve_shape": str(row.get('releve_shape', 'irregular')).strip().lower(),
            "concern_tier": str(row.get('concern_tier', 'green')),
            "overall_dist_score": int(row.get('disturbance_score', 0)),     
            "animals": animal_data 
        })
    return (map_data,)


@app.cell
def _(json, map_data, mo):
    map_data_json = json.dumps(map_data)

    tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    map_html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: sans-serif; 
                background: #f8f9fa; 
                display: flex; 
                flex-direction: row; 
                height: 100vh; 
                width: 100vw;
                overflow: hidden; 
            }}
        
            /* --- LEFT SIDEBAR --- */
            .sidebar {{
                width: 280px;
                background: white;
                border-right: 1px solid #ccc;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 24px;
                z-index: 10;
                box-shadow: 2px 0 8px rgba(0,0,0,0.05);
            }}
            .sidebar h2 {{ font-size: 18px; color: #222; margin-bottom: 8px; }}
            .sidebar p {{ font-size: 12px; color: #666; line-height: 1.4; }}
        
            .radio-group {{ display: flex; flex-direction: column; gap: 12px; }}
            .radio-group label {{
                cursor: pointer; font-size: 14px; font-weight: bold; color: #444; 
                display: flex; align-items: center; gap: 8px;
                padding: 8px; background: #f4f6f8; border-radius: 6px; border: 1px solid #eee;
                transition: background 0.2s;
            }}
            .radio-group label:hover {{ background: #e2e8f0; }}

            .static-legend {{ font-size: 12px; color: #333; line-height: 1.5; }}
            .static-legend h4 {{ font-size: 12px; text-transform: uppercase; margin-bottom: 10px; color: #555; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
            .legend-item {{ display: flex; align-items: center; margin-bottom: 8px; }}
            .legend-color {{ width: 16px; height: 16px; margin-right: 10px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.3); }}

            /* --- RIGHT MAIN CONTENT --- */
            .main-content {{
                flex-grow: 1;
                position: relative;
                background: #e9ecef;
                overflow: hidden; 
            }}
        
            .view-panel {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; }}
            #map-panel {{ display: block; }} 
            #puzzle-panel {{ 
                display: none; 
                align-items: center; 
                justify-content: center; 
            }}
        
            #map-view {{ height: 100%; width: 100%; }}
        
            /* Alaska Puzzle Grid */
            #alaska-grid {{
                display: grid;
                grid-template-columns: repeat(12, 1fr);
                gap: 4px;
                width: 100%;
                max-width: 550px; 
            }}
            .puzzle-piece {{
                aspect-ratio: 1; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);
                transition: transform 0.2s, box-shadow 0.2s; cursor: crosshair;
            }}
            .puzzle-piece:hover {{ transform: scale(1.2); box-shadow: 0 4px 8px rgba(0,0,0,0.3); z-index: 10; }}
            .empty-slot {{ background: transparent; pointer-events: none; }}

            /* Tooltip CSS */
            .custom-plot-marker {{ background: transparent; border: none; }}
            .marker-shape {{ border: 1px solid #222; box-shadow: 0 2px 4px rgba(0,0,0,0.4); opacity: 0.85; transition: transform 0.2s; }}
            .marker-shape:hover {{ transform: scale(1.4); opacity: 1; }}
            .leaflet-tooltip {{ padding: 0; border: none; box-shadow: none; background: transparent; }}
        
            .shared-tooltip-content {{ padding: 10px; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); background: white; color: #333; width: 180px; }}
        
            #global-tooltip {{ position: fixed; pointer-events: none; display: none; z-index: 9999; }}

            .tooltip-header {{ border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-bottom: 6px; text-align: center; }}
            .tooltip-subheader {{ font-size: 11px; margin-bottom: 6px; text-align: center; border-bottom: 1px solid #ddd; padding-bottom: 6px; line-height: 1.4; }}
            .bar-row {{ display: flex; align-items: center; margin-bottom: 4px; font-size: 11px; }}
            .bar-label {{ width: 65px; text-align: left; }}
            .bar-track {{ flex-grow: 1; height: 8px; background: #eee; border-radius: 4px; margin: 0 6px; width: 60px; }}
            .bar-fill {{ height: 100%; background: #607d8b; border-radius: 4px; }}
            .bar-value {{ width: 15px; text-align: right; font-weight: bold; }}
            .human-warning {{ font-size: 11px; text-align: center; padding-top: 4px; color: #d32f2f; font-weight: bold; }}
        </style>
    </head>
    <body>
    
        <div class="sidebar">
            <div>
                <p>Hover over plots to view physical dimensions and local disturbance activity.</p>
            </div>
        
            <div class="radio-group">
                <label><input type="radio" name="viewToggle" value="map" checked> Satellite Map</label>
                <label><input type="radio" name="viewToggle" value="puzzle"> Grid Layout</label>
            </div>

            <div class="static-legend">
                <h4>Concern Tier</h4>
                <div class="legend-item"><div class="legend-color" style="background: #f44336;"></div>High (Red)</div>
                <div class="legend-item"><div class="legend-color" style="background: #ff9800;"></div>Medium (Yellow)</div>
                <div class="legend-item"><div class="legend-color" style="background: #4caf50;"></div>Low (Green)</div>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd; color: #777;">
                    <strong>Size</strong>: Scaled by Area<br>
                    <strong>Shape</strong>: Square, Rectangle, Irregular
                </div>
            </div>
        </div>

        <div class="main-content">
            <div id="map-panel" class="view-panel">
                <div id="map-view"></div>
            </div>
            <div id="puzzle-panel" class="view-panel">
                <div id="alaska-grid"></div>
            </div>
        
            <div id="global-tooltip"></div>
        </div>

        <script>
            const plots = {map_data_json};

            const colorMap = {{
                'red': '#f44336',
                'yellow': '#ff9800',
                'green': '#4caf50'
            }};

            // --- SHARED TOOLTIP GENERATOR ---
            function getTooltipHTML(plot, bgColor) {{
                let lowerContentHTML = "";
                if (plot.dist_type_code === 3) {{
                    lowerContentHTML = `<div class="human-warning">Primary Impact: Anthropogenic</div>`;
                }} else {{
                    if (plot.animals && plot.animals.length > 0) {{
                        lowerContentHTML = plot.animals.map(a => `
                            <div class="bar-row">
                                <div class="bar-label">${{a.animal}}</div>
                                <div class="bar-track"><div class="bar-fill" style="width: ${{a.score * 10}}%"></div></div>
                                <div class="bar-value">${{a.score}}</div>
                            </div>
                        `).join('');
                    }} else {{
                        lowerContentHTML = "<div style='font-size:11px; text-align:center; padding-top:4px; color:#777;'>No animal activity recorded</div>";
                    }}
                }}

                return `
                    <div class="shared-tooltip-content">
                        <div class="tooltip-header"><strong>Plot ${{plot.plot_id}}</strong></div>
                        <div class="tooltip-subheader">
                            Area: ${{plot.releve_area}} m²<br>
                            Tier: <span style="color:${{bgColor}};font-weight:bold;">${{plot.concern_tier.toUpperCase()}}</span><br>
                            <strong>Overall Disturbance: ${{plot.overall_dist_score}}/10</strong>
                        </div>
                        ${{lowerContentHTML}}
                    </div>
                `;
            }}

            // --- INITIALIZE LEAFLET MAP ---
            const map = L.map('map-view').setView([69.845, -155.985], 14);
            L.tileLayer('{tile_url}', {{ attribution: 'Tiles &copy; Esri' }}).addTo(map);

            plots.forEach(plot => {{
                const baseSize = Math.sqrt(plot.releve_area) * 2.5;
                let width = baseSize, height = baseSize, borderRadius = '50%';
            
                if (plot.releve_shape === 'square') borderRadius = '2px';
                else if (plot.releve_shape === 'rectangular') {{ borderRadius = '2px'; width *= 1.4; height *= 0.7; }}

                const bgColor = colorMap[plot.concern_tier] || '#aaaaaa';
                const markerHtml = `<div class="marker-shape" style="width: ${{width}}px; height: ${{height}}px; background: ${{bgColor}}; border-radius: ${{borderRadius}};"></div>`;

                const icon = L.divIcon({{ className: 'custom-plot-marker', html: markerHtml, iconSize: [width, height], iconAnchor: [width/2, height/2] }});

                L.marker([plot.latitude, plot.longitude], {{ icon: icon }})
                    .addTo(map)
                    .bindTooltip(getTooltipHTML(plot, bgColor), {{ direction: 'top', offset: [0, -(height/2)], opacity: 1 }});
            }});


            // --- INITIALIZE PUZZLE VIEW & SMART HOVER ---
            const alaskaShape = [
                [0,0,1,1,1,1,1,1,1,1,0,0],
                [0,1,1,1,1,1,1,1,1,1,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,1,1,1,1,1,1,1,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0],
                [0,1,1,1,1,1,1,1,0,0,0,0],
                [0,1,1,1,0,0,1,1,0,0,1,1],
                [1,1,1,1,0,0,0,0,1,1,1,1],
                [1,1,1,1,1,0,0,0,0,1,1,1] 
            ];

            const gridContainer = document.getElementById('alaska-grid');
            const domTooltip = document.getElementById('global-tooltip');
            let plotIndex = 0;

            // FIXED: Strongly defaults to the right with increased padding
            function updateTooltipPosition(e) {{
                if (domTooltip.style.display === 'none') return;

                const tooltipWidth = domTooltip.offsetWidth;
                const tooltipHeight = domTooltip.offsetHeight;
                const padding = 25; // INCREASED: distance from cursor
                const margin = 10;  // Minimum distance from window edges

                const cx = e.clientX;
                const cy = e.clientY;
                const vw = window.innerWidth;
                const vh = window.innerHeight;

                // Default: Place to the RIGHT of the cursor, centered vertically
                let left = cx + padding;
                let top = cy - (tooltipHeight / 2);

                // Vertical boundaries: if too close to top or bottom, slide along the margins
                if (top < margin) {{
                    top = margin;
                }} else if (top + tooltipHeight > vh - margin) {{
                    top = vh - tooltipHeight - margin;
                }}

                // Horizontal boundaries: if it hits the right edge, flip it completely to the LEFT side
                if (left + tooltipWidth > vw - margin) {{
                    left = cx - tooltipWidth - padding;
                }}

                domTooltip.style.left = left + 'px';
                domTooltip.style.top  = top  + 'px';
            }}

            for (let r = 0; r < alaskaShape.length; r++) {{
                for (let c = 0; c < alaskaShape[r].length; c++) {{
                    const cell = document.createElement('div');
                
                    if (alaskaShape[r][c] === 1 && plotIndex < plots.length) {{
                        const plot = plots[plotIndex];
                        const bgColor = colorMap[plot.concern_tier] || '#aaaaaa';
                    
                        cell.className = 'puzzle-piece';
                        cell.style.backgroundColor = bgColor;
                    
                        cell.addEventListener('mouseenter', (e) => {{
                            domTooltip.innerHTML = getTooltipHTML(plot, bgColor);
                            domTooltip.style.display = 'block';
                            updateTooltipPosition(e);
                        }});
                        cell.addEventListener('mousemove', updateTooltipPosition);
                        cell.addEventListener('mouseleave', () => {{
                            domTooltip.style.display = 'none';
                        }});

                        plotIndex++;
                    }} else {{
                        cell.className = 'empty-slot';
                    }}
                    gridContainer.appendChild(cell);
                }}
            }}

            // --- TOGGLE LOGIC ---
            const mapPanel = document.getElementById('map-panel');
            const puzzlePanel = document.getElementById('puzzle-panel');
            const radios = document.querySelectorAll('input[name="viewToggle"]');

            radios.forEach(radio => {{
                radio.addEventListener('change', (e) => {{
                    if (e.target.value === 'map') {{
                        puzzlePanel.style.display = 'none';
                        mapPanel.style.display = 'block';
                        setTimeout(() => map.invalidateSize(), 10); 
                    }} else {{
                        mapPanel.style.display = 'none';
                        puzzlePanel.style.display = 'flex'; 
                    }}
                }});
            }});
        </script>
    </body>
    </html>"""

    mo.Html(f"""
    <div style="border-radius:8px; border:2px solid #ccc; overflow:hidden;">
        <iframe
            srcdoc="{map_html.replace('"', '&quot;')}"
            style="width:100%; height:650px; border:none; display:block;"
            scrolling="no"
            sandbox="allow-scripts allow-same-origin"
        ></iframe>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
