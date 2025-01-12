cdef class Player:
    cdef public str id
    cdef public double rating
    cdef public int matches_played
    cdef public list rating_history
    cdef public int streak