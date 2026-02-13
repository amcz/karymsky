#import logging

import matplotlib

#logger = logging.getLogger(__name__)

def get_qva_colors_rgb():
        """@brief Get standard QVA (Volcanic Ash Advisory) colors
        @return List of tuples with (linecolor, hexcolor) for light blue, orange, red, purple
        """
        # QVA colors in RGB
        colors = [
            (207, 199, 198),   # Light gray
            (105, 105, 105),   # Dark gray (added)
            (160, 210, 255),   # Light blue
            (255, 153, 0),     # Orange
            (255, 40, 0),      # Red
            (170, 0, 170),     # Purple
            (227, 15, 242)     # Pink
        ]
        alpha = 200  # C8 in decimal
        def rgb_to_rgba_hex(r, g, b, a=alpha):
            return f"#{r:02X}{g:02X}{b:02X}{a:02X}"
        clist = [rgb_to_rgba_hex(r, g, b) for (r, g, b) in colors]
        return clist


class ConcplotColors:
    def __init__(self):
        colorhash = {}
        colorhash["yellow"] = "242236051"
        colorhash["orange"] = "235137052"
        colorhash["red"] = "161024014"
        colorhash["blue"] = "070051242"
        colorhash["green"] = "147219121"
        colorhash["magenta"] = "194056143"
        colorhash["purple"] = "107023156"
        colorhash["cyan"] = "075201199"
        colorhash["grey"] = "150150150"
        colorhash["tan"] = "163145131"
        self.colorhash = colorhash

    def get(self, color):
        return self.colorhash[color]


def color2kml(cstr):
    """
    python used RGB while KML expects BGR.
    """
    return cstr[0:2] + cstr[-2:] + cstr[4:6] + cstr[2:4]

class ColorMaker:
    def __init__(self, cname, nvals, ctype="hex", transparency="C8"):
        """
        cname : name of matplotlib colormap
        nvals : number of color values
        ctype : if 'hex' returns 8 digit hexidecimal with
                transparancy.
        transparency : str: transparency value to use in hexidecimal.
        """
        self.transparency = transparency
        self.clist = []      # list of nvals colors equally spaced througout the colormap
        self.ctype = ctype
        self.get_cmap(cname, nvals)

    def __call__(self):
        """
        Returns:
        list of nvals colors equally spaced throughout the colormap.
        and in hexidecimal format.
        """
        return self.clist

    def rgb_to_hex(self, rgb):
        """
        convert from rgb to hexidecimal.
        """

        def subfunc(val):
            rval = hex(int(val * 255)).replace("0x", "").upper()
            if len(rval) == 1:
                rval = "0" + rval
            return rval

        hval = [subfunc(x) for x in list(rgb)]
        if self.transparency:
            return "{}{}{}{}".format(self.transparency, hval[0], hval[1], hval[2])
        else:
            return "{}{}{}".format(hval[0], hval[1], hval[2])

    def get_cmap(self, cname="viridis", nvals=10):
        cmap = matplotlib.cm.get_cmap(cname)
        cvals = cmap.N
        cspace = int(cvals / nvals)
        if nvals%2 > 0: cspace+=1
        if self.ctype == "hex":
            self.clist = [self.rgb_to_hex(cmap(x)) for x in range(0, cvals, cspace)]
        else:
            self.clist = [cmap(x) for x in range(0, cvals, cspace)]
          

        # for iii in range(0,cvals,cspace):
        #    if ctype == 'hex':
        #        self.clist.append(self.rgb_to_hex(cmap(iii)))
        #    else:
        #        self.clist.append(cmap[iii])



def get_vaa_colors():
    """
    @brief Returns list of RGB colors used in Volcanic Ash Advisory graphics
    @return List of RGB tuples for: light blue, orange, red, purple
    """
    return [
        (160, 210, 255),  # Light blue
        (255, 153, 0),    # Orange
        (255, 40, 0),     # Red
        (170, 0, 170)     # Purple
    ]

def get_vaa_colors_hex():
    """
    @brief Returns list of hex colors used in Volcanic Ash Advisory graphics
    @return List of hex strings for: light blue, orange, red, purple
    """
    return [
        "#A0D2FF",  # Light blue
        "#FF9900",  # Orange
        "#FF2800",  # Red
        "#AA00AA"   # Purple
    ]

def get_vaa_colors_kml():
    """
    @brief Returns list of KML colors used in Volcanic Ash Advisory graphics
    @return List of KML strings for: light blue, orange, red, purple
    """
    clrs = get_vaa_colors()
    return [color2kml(x) for x in clrs]