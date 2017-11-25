project (mathlib)

set(HEADERS DirichletProblem.h
			DirichletProblem.hpp
			GridOperations.h
			GridOperations.hpp
	)
	
set(SOURCE Functions.cpp
			GridOperations.cpp)

add_library(mathlib ${SOURCE} ${HEADERS})
