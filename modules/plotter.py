import mpld3

def HTMLPlotter(fig, html_path):
    css = """
    <style>
       html, body {
          overflow: hidden;
          height: 100%; 
          padding: 0;
          margin: 0;
       }
       div {
          justify-content: center;
          align-items: center;
          text-align: center;
          display: flex;
          height: 100vh;
          width: 100%;
       }

       svg {
         max-width: 100%;
         max-height: 100%;
      }
    </style>
    """
    with open(html_path, 'w') as f:
        f.write(css + mpld3.fig_to_html(fig))
    return html_path
