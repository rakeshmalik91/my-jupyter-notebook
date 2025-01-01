\version "2.24.2"

\header {
  title = "Sci-Fi Music"
  composer = "Generated with LilyPond"
}

\score {
  \relative c' {
    \key c \minor
    \time 5/4
    \tempo "Mysterious and futuristic" 4 = 80

    % Opening theme
    c8 g' bes g | 
    c4.~ c8 d e f | 
    g2. f8 e | 
    d4. g,8 c |

    % Alien-like motif
    ees16 f g aes g f ees d |
    c8 r8 ees16 f g ees c g |

    % Build tension
    g,8 c' d ees f g aes g |
    f4.~ f8 ees d c |

    \bar "|." % Ending mark
  }

  \layout { }
  \midi { } % This generates the MIDI file
}
