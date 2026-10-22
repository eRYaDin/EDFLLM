def summarize(report):
    lines=['Perbandingan deskriptif EEG A dan B',
           f"Durasi: {report['duration_s']:g} s | Kanal: {', '.join(report['channels'])}",
           f"Sampling analisis: {report['settings']['fs']:g} Hz | amplitudo uV; power uV²."]
    for channel in report['channels']:
        rows=[r for r in report['comparison'] if r['channel']==channel and r['A'] is not None]
        for row in rows:
            if row['metric'] in ('rms_uv','power_alpha_uv2','power_beta_uv2'):
                lines.append(f"{channel} {row['metric']}: A={row['A']:.4g}, B={row['B']:.4g}, B-A={row['B_minus_A']:+.4g}")
    lines+=['','Catatan:']+['• '+w for w in report['warnings']]
    return '\n'.join(lines)
